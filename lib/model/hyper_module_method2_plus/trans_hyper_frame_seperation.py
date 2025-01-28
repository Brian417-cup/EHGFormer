import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__)))
sys.path.append(osp.join(osp.dirname(__file__), '../../..'))

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from einops import rearrange
from lib.model.hyper_module_method2_plus.trunc_normal import trunc_normal_, _no_grad_trunc_normal_
from lib.model.hyper_module_method2_plus.drop_path import DropPath
from lib.model.hyper_module_method2_plus.mlp import Mlp

import numpy as np
# import numpy.typing as npt
from functools import partial

# for abladation study
from lib.model.hyper_module_method2_plus import abladation_study


############################

class HyperEdge(nn.Module):
    '''
    Idea comes from https://arxiv.org/pdf/2211.09590.pdf
    '''

    def __init__(self, in_c: int, out_c: int, joint_label: List[List[int]] = None, num_joint: int = 17,
                 hyper_head: int = 1, need_proj=True, spatial_mode='v', multi_merge_mode='mean'):
        super(HyperEdge, self).__init__()

        # Not implement the vc spatial mode compeletely
        assert spatial_mode == 'v', "Not implement the 'vc' == spatial_mode compeletely"
        assert multi_merge_mode in ('none', 'mean', 'max', 'weight_mean'), "Current merge way doesn't support"

        self.joint_label = joint_label
        self.in_c = in_c
        self.out_c = out_c
        self.num_joint = num_joint
        self.spatial_mode = spatial_mode
        self.multi_merge_mode = multi_merge_mode
        if joint_label is not None:
            self.way = 'hard'
            self.hyper_head = len(joint_label)
            assert self.hyper_head == hyper_head, 'Desired hyper head is not equal to the input sequence length about joint labels'
            if hyper_head == 1:
                self.joint_label = joint_label[0]

            self._build_init_hyper_matrix()
        else:
            self.way = 'soft'
            self.hyper_head = hyper_head
            self._build_init_hyper_matrix()

        # proj for hyper edge in every feature all channels
        if need_proj:
            # self.pro_linears = [nn.Linear(in_c, out_c, bias=False)] * self.hyper_head
            self.pro_linears = [nn.Linear(in_c, out_c, bias=False)]
        else:
            # self.pro_linears = [nn.Identity()] * self.hyper_head
            self.pro_linears = [nn.Identity()]
        # self.pro_linears = [nn.Conv2d(in_c, out_c, 1, bias=False)] * self.hyper_head
        self.pro_linears = nn.ModuleList(self.pro_linears)

        # decide multi hyper matrix merge way
        if self.hyper_head == 1:
            self.merge_weight = nn.Parameter(torch.ones(1, requires_grad=False), requires_grad=False)
        else:
            raise NotImplementedError

    def _build_init_hyper_matrix(self):
        '''
        利用one-hot编码来构建 V*E的超边矩阵,用于超图矩阵的 初始化 init 时调用
        Set static joint_label and hypermatrix or only get dynamic hypermatrix.
        If static joint label, hyper matrix can be directly binarazied from joint label, and hyper matrix wihout any grad.
        If dynamic joint label, hyper matrix is paramaterized in random way and need grad, it don't have joint label in initialization.
        :return: [V,E]
        '''
        if self.way == 'hard':
            self.joint_label = nn.Parameter(torch.tensor(self.joint_label), requires_grad=False)
            # new version
            # self.hyper_matrix = nn.Parameter(F.one_hot(self.joint_label, num_classes=self.num_joint).float(),
            #                                  requires_grad=False)
            # old version (Used for visualization currently)
            self.hyper_matrix = nn.Parameter(F.one_hot(self.joint_label).float(),
                                             requires_grad=False)
            # self.hyper_matrix = None

            # print(self.joint_label.shape)
            # print(self.hyper_matrix.shape)
        # In the dynamic mode, the hypermatrix and joint_label should be caculate dynamic every epoch
        # rather than in this init epoch
        else:
            # hyper_matrix [0,1]
            # 随机初始化超图矩阵
            # [hyper_head,V,V>=E]
            if self.hyper_head > 1:
                raise NotImplementedError
            # [V,V>=E]
            else:
                if self.spatial_mode == 'v':
                    self.hyper_matrix = nn.Parameter(torch.randn((self.num_joint, self.num_joint)),
                                                     requires_grad=True)
                else:
                    raise NotImplementedError

    def _get_hyper_edge_degree(self, dealt_hypermatrix):
        '''
        用于获取超图矩阵的第一种方法
        :param dealt_hypermatrix: [V,E<=V]
        :return: [E,E]
        '''
        per_edge_joint_cnt = torch.sum(dealt_hypermatrix, dim=-2, keepdim=False)
        # print(per_joint_edge_cnt.shape)
        # |E|
        edge_cnt = dealt_hypermatrix.shape[-1]

        edge_degree_matrix = torch.zeros(edge_cnt, edge_cnt) + 1e-6
        for i in range(edge_cnt):
            edge_degree_matrix[i, i] = per_edge_joint_cnt[i]
        return edge_degree_matrix

    def get_hyper_feature(self, x, joint_label, dealt_hypermatrix, no=0):
        '''
        这边和HDFormer有异曲同工之处，用点特征表示超边特征，再把超边的特征融合到点特征中
        :param x: [N*T,V,C]
        :param joint_label: [V] or [hyper_head,V]
        :param dealt_hypermatrix: [V,E<=V]
        :return:  f_e [N*T,V,C]
        '''
        # one way to converge hyperedges into joints

        _, V, C = x.shape

        # for single hyper head, it can be optimized in a more effeicient way to converge hyperedges into joints
        if self.hyper_head == 1:
            # Init hyperedge acquire way (For train and evaluation)
            if abladation_study.args.deployment == '':
                # [N*T,V,C] -> [N*T,C,V]
                x = x.permute(0, 2, 1)

                # [V,E<=V]
                label = F.one_hot(joint_label).float().to(x.device)
                # [N*T,C,V] @ [V,E<=V] = [N*T,C,E<=V]
                z = x @ (label / label.sum(dim=0, keepdim=True))

                # [E<=V,N*T,C]
                z = self.pro_linears[no](z.permute(0, 2, 1)).permute(1, 0, 2)
                # [N*T,V,C]
                f_e = z[joint_label].permute(1, 0, 2)
            # lower spped hyperedge acquire way(For deployment)
            else:
                hyper_edge_degree = self._get_hyper_edge_degree(dealt_hypermatrix)
                # [N*T,V,C]
                f_e = \
                    dealt_hypermatrix @ self.custom_matrix_inverse_for_deployment(hyper_edge_degree).to(x.device) @ \
                    dealt_hypermatrix.transpose(-1, -2) @ x
                # in [N*T,V,C] do linear
                f_e = self.pro_linears[no](f_e)

        # else it only can be converge hyperedges into joints in a slower way
        else:
            # hyper_edge_degree = self._get_hyper_edge_degree(dealt_hypermatrix)
            # # [N*T,V,C]
            # f_e = dealt_hypermatrix @ torch.linalg.inv(hyper_edge_degree).to(x.device) @ dealt_hypermatrix.T @ x
            # # in [N*T,V,C] do linear
            # f_e = self.pro_linears[no](f_e)
            raise NotImplementedError

        return f_e

    def get_forward_joint_label_and_hypermatrix(self):
        '''
            In the forward process to caculate hypermatrix and joint_label
            in hard or soft way.
            :return: [V] or [hyper_head,V], [V,V>=E] or [hyper_head,V,V>=E]
            '''
        # single hyperhead
        if self.hyper_head == 1:
            # If hard, the joint_label and hypermatrix are static , they are stored in initialization.
            if self.way == 'hard':
                return self.joint_label, self.hyper_matrix
            # else in the soft mode, every forward the joint_label and hypermatrix
            # are dynamic
            else:
                assert NotImplementedError
        # hyperhead >1
        else:
            assert NotImplementedError

    def get_eval_joint_label(self):
        '''
        Get joint label in eval stage.
        If it's hard, it can be directly obtained by stored joint label variable
        Else, it is infered by hyper matrix.
        :return: [hyper_head,V,E<=V]
        '''
        if self.way == 'hard':
            if self.hyper_head == 1:
                return torch.unsqueeze(self.joint_label, dim=0)
            else:
                assert NotImplementedError
        else:
            assert NotImplementedError

    def custom_matrix_inverse_for_deployment(self, src_matrix):
        '''
        Custom matrix inverse for torch.linalg.inv(), which is not supported for init onnx export.
        The result is same with torch.linalg.inve()
        :param src_matrix: [V,E]
        :return:
        '''

        def cof1(M, index):
            zs = M[:index[0] - 1, :index[1] - 1]
            ys = M[:index[0] - 1, index[1]:]
            zx = M[index[0]:, :index[1] - 1]
            yx = M[index[0]:, index[1]:]
            s = torch.cat((zs, ys), axis=1)
            x = torch.cat((zx, yx), axis=1)
            return torch.linalg.det(torch.cat((s, x), axis=0))

        def alcof(M, index):
            return pow(-1, index[0] + index[1]) * cof1(M, index)

        def adj(M):
            result = torch.zeros((M.shape[0], M.shape[1]))
            for i in range(1, M.shape[0] + 1):
                for j in range(1, M.shape[1] + 1):
                    result[j - 1][i - 1] = alcof(M, [i, j])
            return result

        return 1.0 / torch.linalg.det(src_matrix) * adj(src_matrix)

    def forward(self, x):
        '''

        :param x: [N*T,V,C]
        :return: Eaug in thesis (combine the hyperedge feature into joint feature)
                [N*T,V,C]
        '''
        _, V, C = x.shape

        joint_label, hyper_matrix = self.get_forward_joint_label_and_hypermatrix()

        # [N*T,V,C]
        f_e = self.get_hyper_feature(x, joint_label, hyper_matrix, no=0)
        return f_e, joint_label, hyper_matrix


class HyperAttention(nn.Module):
    def __init__(self, dim_in, dim, A, graph_mode='none',
                 num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0, visualize=False):
        '''

        :param dim_in: Used in qkv in channels
        :param dim: Used in qkv out channels and embedding channel
        :param A:
        :param num_heads:
        :param qkv_bias:
        :param qk_scale:
        :param attn_drop:
        :param proj_drop:
        :param insert_cls_layer:
        :param pe:
        :param num_point:
        :param outer:
        :param layer:
        :param visualize: if need visualize such as attention map,it should be True, default mode is False
        '''
        super().__init__()
        self.num_heads = num_heads if dim > num_heads else 1
        self.dim_in = dim_in
        self.dim = dim
        head_dim = self.dim // self.num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer

        # 求最短路径
        # h1 = A.sum(0)
        # h1[h1 != 0] = 1
        # h = [None for _ in range(num_point)]
        # h[0] = np.eye(num_point)
        # h[1] = h1
        # self.hops = 0 * h[0]
        # for i in range(2, num_point):
        #     h[i] = h[i - 1] @ h1.transpose(0, 1)
        #     h[i][h[i] != 0] = 1
        #
        # for i in range(num_point - 1, 0, -1):
        #     if np.any(h[i] - h[i - 1]):
        #         h[i] = h[i] - h[i - 1]
        #         self.hops += i * h[i]
        #     else:
        #         continue
        #
        # self.hops = torch.tensor(self.hops).long()

        # self.hops = self.get_shortest_path_matrix(A)

        # 构建位置编码查询表
        # self.rpe = nn.Parameter(torch.zeros((self.hops.max() + 1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))

        self.kv = nn.Linear(dim_in, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim_in, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # self.proj = nn.Conv2d(dim, dim, 1, groups=6)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

        # For visualize stage, this value is not None
        self.visualize = visualize
        self.attn_map = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def get_shortest_path_matrix(self, A: Union[npt.NDArray[np.uint8], torch.Tensor]) -> torch.Tensor:
    def get_shortest_path_matrix(self, A: Union[np.array, torch.Tensor]) -> torch.Tensor:
        '''
        Get shortest path matrix
        :param A: [V,V] or [H,V,V]
        :return: [V,V], every shortest[i,j]=k means from i to j,there is a shortest path that distanced k.
        '''
        # max hop is num_joint
        k_hop = num_joint = A.shape[-1]
        # N= A.shape[0]
        # [H,V,V] -> [V,V]
        if len(A.shape) == 3:
            H = A.sum(0)
        else:
            H = A
        H[H != 0] = 1

        k_hops_A = [None for _ in range(k_hop)]
        k_hops_A[0] = np.eye(k_hop, dtype=np.uint8)
        k_hops_A[1] = H

        for i in range(2, k_hop, 1):
            k_hops_A[i] = k_hops_A[i - 1] @ H.transpose(0, 1)
            k_hops_A[i][k_hops_A[i] != 0] = 1

        shortest_path_k_A = [None for _ in range(k_hop)]
        shortest_path_k_A[0] = np.eye(num_joint, dtype=np.uint8)
        # shortest_path_k_A[0] = np.eye(k_hop, dtype=np.uint8)
        for i in range(k_hop - 1, 0, -1):
            shortest_path_k_A[i] = k_hops_A[i] - k_hops_A[i - 1]
            shortest_path_k_A[i][shortest_path_k_A[i] != 0] = 1

        # final_combine_sum = np.zeros((k_hop, k_hop), dtype=np.uint8)
        final_combine_sum = np.zeros((num_joint, num_joint), dtype=np.uint8)

        for k, spd_a_hop_k in enumerate(shortest_path_k_A):
            final_combine_sum += spd_a_hop_k * k

        return torch.tensor(final_combine_sum).long()

    def forward(self, x, e):
        '''

        :param x: [N*T,V,C]
        :param e: [N*T,V,C]
        :return: [N*T,V,C]
        '''
        # N, C, T, V = x.shape
        _, V, C = x.shape
        # [2,N*T,h,V,C//h]
        kv = self.kv(x).reshape(-1, 2, self.num_heads, self.dim // self.num_heads, V).permute(1, 0, 2, 4, 3)
        # [N*T,h,V,C//h]
        k, v = kv[0], kv[1]

        ## n*t h v c
        q = self.q(x).reshape(-1, self.num_heads, self.dim // self.num_heads, V).permute(0, 1, 3, 2)
        # [N*T,h,V,C//h]
        e_k = e.reshape(-1, self.num_heads, self.dim // self.num_heads, V).permute(0, 1, 3, 2)

        #
        # b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
        # b = torch.einsum("bhnc, nmhc->bhnm", q, k_r)
        #
        # c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
        # c = torch.einsum("bhnc, bhmc->bhnm", q, e_k)
        c = q @ e_k.transpose(-1, -2)

        # d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)
        d = torch.einsum("hc, bhmc->bhm", self.w1, e_k).unsqueeze(-2)

        a = q @ k.transpose(-2, -1)

        # [N*T,h,V,V]
        # attn = a + b + c + d
        # attn = a + c + d
        # In the hyper method2, we inject the edge feature in the joint without joint-by-joint again
        attn = 0
        if abladation_study.j2e_main:
            attn += c
        if abladation_study.j2e_bias:
            attn += d

        # Not for ablation study
        # attn = c + d

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        # This attn heatmap is only for the test visualization
        if self.visualize:
            self.attn_map = attn.clone()

        attn = self.attn_drop(attn)

        # [N*T,h,V,C//h]
        x = attn @ v

        # [N*T,h,C//h,V] -> [N*T,C,V] -> [N*T,V,C]
        x = x.transpose(2, 3).reshape(-1, self.dim, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x


class HyperAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, graph_mode='none',
                 mlp_hidden_dim=1024, num_heads=6,
                 pe_proj=True, num_joint=17, layer=0, add_skip_connection=True, qkv_bias=False,
                 qk_scale=None, drop=0., joint_label: List[List[int]] = None,
                 attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 insert_cls_layer=0, spatial_mode='v', hyper_head: int = 1, multi_merge_mode='mean', visualize=False):
        '''

        :param in_channels:
        :param out_channels:
        :param A: [3,V,V]
        :param mlp_hidden_dim:
        :param stride:
        :param num_heads:
        :param residual:
        :param kernel_size:
        :param dilations:
        :param pe_proj: Need hyper channels project when getting the hyperedge feature to joint feature
        :param num_joint:
        :param layer: Not used in temp.
        :param add_skip_connection:
        :param qkv_bias:
        :param qk_scale:
        :param drop:
        :param joint_label:
        :param attn_drop:
        :param drop_path:
        :param act_layer:
        :param norm_layer:
        :param insert_cls_layer: Not used in temp.
        :param visualize: if need visualize such as attention map,it should be True, default mode is False
        '''
        super(HyperAttnBlock, self).__init__()
        # In the MHFormer Spatial Encoder, the qkv in in_channels and out_channels are same
        assert in_channels == out_channels, 'In MixSTE mode, the in channels and out channels should be same.'

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(out_channels)

        self.in_c = in_channels
        self.out_c = out_channels
        self.mlp_hidden = mlp_hidden_dim
        self.mlp = Mlp(in_features=out_channels, hidden_features=mlp_hidden_dim)

        self.add_skip_connection = add_skip_connection
        self.num_joint = num_joint

        if spatial_mode == 'v':
            self.hyper_edge = HyperEdge(in_channels, out_channels, joint_label=joint_label, num_joint=num_joint,
                                        need_proj=pe_proj, spatial_mode='v', hyper_head=hyper_head,
                                        multi_merge_mode=multi_merge_mode)
        else:
            assert NotImplementedError

        # In the MHFormer Spatial Encoder, the qkv in in_channels and out_channels are same
        self.attn = HyperAttention(in_channels, out_channels,
                                   A, graph_mode=graph_mode,
                                   num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe_proj, num_point=num_joint,
                                   layer=layer, visualize=visualize)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.in_c != self.out_c:
            self.skip_proj = nn.Linear(in_channels, out_channels, bias=False)

        self.pe_proj = pe_proj
        # self.pe_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x, T=243):
        '''

        :param x: [N*T,V,C]
        :return: [N*T,V,C] , [V] , [V,E<=V]
        '''
        # [N*T,V,C] ,[hyper_head,V] or [V], [V,E<=V]
        f_e, joint_label, hyper_matrix = self.hyper_edge(x)

        # 这块HyperFormer中是没有第二个残差结构的，这里的第二个残差步骤是参考了MHFormer的
        if self.add_skip_connection:
            if self.in_c != self.out_c:
                x = self.skip_proj(x) + self.drop_path(
                    self.attn(self.norm1(x), f_e))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), f_e))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.drop_path(self.attn(self.norm1(x), f_e))
            x = self.drop_path(self.mlp(self.norm2(x)))

        #  [N,C,T,V] -> [N*T,V,C]
        # x = rearrange(x, 'n c t v -> (n t) v c', t=T, v=self.num_joint, c=self.out_c)
        return x, joint_label, hyper_matrix


def test_paramter():
    a = nn.Parameter(torch.ones(5))
    a = F.softmax(a)
    print(a)


def test_sum():
    a = torch.arange(1, 7).view(2, 3)
    s1 = torch.sum(a, dim=-2)
    s2 = torch.sum(a, dim=-1)
    print(s1)
    print(s2)


if __name__ == '__main__':
    N, T, V, C = 2, 243, 17, 256
    num_head = 8
    # assert C % num_head == 0, '通道维度必须被9给整除'
    x = torch.rand(N * T, V, C)
    A = np.random.random([3, V, V])
    joint_label = None
    hyper_head = 1
    multi_merge_mode = 'none'

    hyper_attn_block = HyperAttnBlock(in_channels=C, out_channels=C, A=A, num_heads=num_head, num_joint=V, pe_proj=True,
                                      joint_label=joint_label, spatial_mode='v', hyper_head=hyper_head,
                                      multi_merge_mode=multi_merge_mode)
    y, joint_label, hyper_matrix = hyper_attn_block(x, T=T)
    print(y.shape)
