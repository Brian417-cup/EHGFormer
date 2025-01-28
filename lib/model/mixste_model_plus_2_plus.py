import logging
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))

from functools import partial
from einops import rearrange
import numpy as np
# import numpy.typing as npt
import torch
import torch.nn as nn
from lib.model.hyper_module_method2_plus.trans_temporal import Block as Temporal_Block
from graph.graph import Graph
from typing import Union


class MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 spatial_pos_type='common',
                 hyper_opt=None, graph_mode='none', hrnet=False, visuzlie=False,
                 distill_middle=False, middle_no_list=[]):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            spatial_pos_type (str): choices of arguments in pos_type, which decider the type of spatial positional embedding
            hyper_opt: arguments from cmd
            visualize (bool): if need visualize such as attention map,it should be True, default mode is False
            distill_middle (bool): if use distill in middle layer, you may need the middle tokens
            middle_no_list (list): if use distill middle, it will record corresponding no
        """
        super().__init__()

        assert hyper_opt.spatial_mode in ['common', 'hyper',
                                          'hyper_frame_seperation',
                                          'hyper_dependent_common'], 'Spatial mode should be in supported way.'
        self.opt = hyper_opt
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3  #### output dimension is num_joints * 3

        ### Graph Load
        if self.opt.dataset in ('h36m', 'pw3d'):
            if hrnet:
                from graph.skeleton import get_skeleton_h36m_hrnet
                get_skeleton = get_skeleton_h36m_hrnet
            else:
                from graph.skeleton import get_skeleton_h36m
                get_skeleton = get_skeleton_h36m
        elif self.opt.dataset == 'mpi-inf-3dhp' or self.opt.dataset.startswith('mpi'):
            from graph.skeleton import get_skeleton_mpi_inf_3dhp
            get_skeleton = get_skeleton_mpi_inf_3dhp
        elif self.opt.dataset.startswith('ntu'):
            # for ntu-rgb+d, the keypoints of skeletons has been converted to h36m format
            from graph.skeleton import get_skeleton_h36m
            get_skeleton = get_skeleton_h36m
        else:
            raise NotImplementedError

        skeleton = get_skeleton()
        if graph_mode == 'none':
            graph = Graph(skeleton=skeleton, strategy='foa', max_hop=1, dilation=1)
        elif graph_mode == 'uniform':
            graph = Graph(skeleton=skeleton, strategy='uniform', max_hop=1, dilation=1)
        else:
            raise NotImplementedError
        # self.A: npt.NDArray = graph.A
        self.A = graph.A
        self.num_joint = self.A.shape[-1]

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # This mhformer_module is especially for the common spatial + hyper spatial
        # currently, only this branch is useful
        if hyper_opt.spatial_mode == 'common' or hyper_opt.spatial_mode == 'hyper_dependent_common':
            if spatial_pos_type == 'common':
                self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio), requires_grad=True)
            else:
                raise NotImplementedError
            # elif spatial_pos_type == 'spd':
            #     hops = self._get_shortest_path_matrix(self.A)
            #     self.Spatial_pos_embed = nn.Parameter(torch.zeros((self.hops.max() + 1, embed_dim)))
        else:
            assert NotImplementedError
            # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio), requires_grad=False)

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim), requires_grad=True)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        # 下面这两个基础块都是在 C=1024 中完成的
        if hyper_opt.spatial_mode == 'common':
            from myself_test.mixste_insert.module_method2.trans_spatial import Block as Spatial_Block
            self.STEblocks = nn.ModuleList([
                Spatial_Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    mode=hyper_opt.spatial_mode)
                for i in range(depth)])
        elif hyper_opt.spatial_mode == 'hyper':
            from myself_test.mixste_insert.module_method1.trans_hyper import HyperAttnBlock as Spatial_Block
            self.STEblocks = nn.ModuleList([
                Spatial_Block(
                    in_channels=embed_dim_ratio, out_channels=embed_dim_ratio, num_joint=num_joints,
                    num_heads=num_heads, mlp_hidden_dim=int(mlp_ratio * embed_dim_ratio), qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, A=self.A,
                    joint_label=hyper_opt.joint_label,
                    spatial_mode='v', hyper_head=hyper_opt.hyper_head,
                    multi_merge_mode=hyper_opt.hyper_multi_merge_mode)
                for i in range(depth)])
        elif hyper_opt.spatial_mode == 'hyper_frame_seperation':
            from myself_test.mixste_insert.module_method1.trans_hyper_frame_seperation import \
                HyperAttnBlock as Spatial_Block
            self.STEblocks = nn.ModuleList([
                Spatial_Block(
                    in_channels=embed_dim_ratio, out_channels=embed_dim_ratio, num_joint=num_joints,
                    num_heads=num_heads, mlp_hidden_dim=int(mlp_ratio * embed_dim_ratio), qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, A=self.A,
                    joint_label=hyper_opt.joint_label,
                    spatial_mode='v', hyper_head=hyper_opt.hyper_head,
                    multi_merge_mode=hyper_opt.hyper_multi_merge_mode)
                for i in range(depth)])
        elif hyper_opt.spatial_mode == 'hyper_dependent_common':
            # And the hyper_dependent_common is only support for hyper_frame_seperation
            from lib.model.hyper_module_method2_plus.trans_common_hyper_sequence_spatial import \
                CommonAndHyperSpatialTrans as Spatial_Block
            self.STEblocks = nn.ModuleList([
                Spatial_Block(embed_dim_ratio=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, drop_rate=dpr[i],
                              norm_layer=norm_layer, A=self.A, graph_mode=graph_mode, num_joints=num_joints,
                              hyper_opt=hyper_opt,
                              visuzlie=visuzlie)
                for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Temporal_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False,
                changedim=False, currentdim=i + 1, depth=depth, mode='temporal', visualize=visuzlie)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # 回归头，作为最终输出的通道维度使得C_out=3,最终的输出尺寸为 [N,T,V,C'=3]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # visualize when it is in the test stage
        self.visualize = visuzlie
        self.joint_label_by_depth = []
        self.hyper_matrix_by_depth = []

        # distill when it need
        self.distill_middle = distill_middle
        #######################################################
        # distillation stage
        # init distillation to store middle feature tokens
        if self.distill_middle:
            self.middle_temporal = []
            self.middle_spatial = []
            # currently, we merely distill middle feature
            self.middle_temporal_ids = middle_no_list
            self.middle_spatial_ids = middle_no_list
        #######################################################

    def _get_shortest_path_matrix(self, A: Union[np.array, torch.Tensor]) -> torch.Tensor:
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

    def STE_forward(self, x, no=0):
        '''
        在forward中先调用这个函数(针对第一次的空间流输入)
        构建 (V,C')@(V,C')T 中V个元素之间的相关性，其中在时间维度上，对不同的时刻是独立建模的，也就是说，不同时刻都要计算相关性,
        这也就意味着时间维度是在计算之前并入了batch_size这个维度中，也就是 N*T
        Args:
            x:

        Returns:

        '''
        # [N,T,V,C]
        b, f, n, c = x.shape
        # [N,T,V,C] -> [N*T,V,C]
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        # [N*T,V,C']
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        if self.opt.spatial_mode == 'common':
            x = blk(x, T=f)
        else:
            if self.opt.j2e:
                x, joint_label, hyper_matrix = blk(x, T=f)
                if self.visualize:
                    self.joint_label_by_depth.append(joint_label.clone())
                    self.hyper_matrix_by_depth.append(hyper_matrix.clone())
            else:
                x = blk(x, T=f)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)

        # for distillation (Used for spatial distill middle currently)
        if self.distill_middle and (no in self.middle_spatial_ids):
            self.middle_spatial.append(x)

        # [N*T,V,C'] -> [N*V,T,C']
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x, no=0):
        '''
        在forward中第二步调用这个模块(针对第一次的时间流输入)
        构建 (T,C')@(T,C')T 中T个元素的相关性，其中在点这个空间维度上,对不同点要求独立做相关性的建模运算,
        这也就意味着空间维度在计算相关性前提前斌入了batch_size这个维度中
        Args:
            x:

        Returns:

        '''
        # [N * V, T, C']
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _ = x.shape
        x += self.Temporal_pos_embed[:, :f, :]
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)

        # for distillation
        if self.distill_middle and (no in self.middle_temporal_ids):
            self.middle_temporal.append(x)

        return x

    def ST_foward(self, x):
        '''
        针对后面几次的空间-时间流输入，
        这里需要注意的是，位置编码信息仅在前面的第一次中加入，这里都不加入位置编码信息
        Args:
            x:

        Returns:

        '''
        # [N,T,V,C']
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            # [N,T,V,C'] -> [N*T,V,C']
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            if self.opt.spatial_mode == 'common':
                x = steblock(x, T=f)
            else:
                if self.opt.j2e:
                    x, joint_label, hyper_matrix = steblock(x, T=f)

                    if self.visualize:
                        self.joint_label_by_depth.append(joint_label.clone())
                        self.hyper_matrix_by_depth.append(hyper_matrix.clone())
                else:
                    x = steblock(x, T=f)

            x = self.Spatial_norm(x)

            # for distillation (Used for spatial distill middle currently)
            if self.distill_middle and (i in self.middle_spatial_ids):
                self.middle_spatial.append(x)

            # [N*T,V,C'] -> [N*V,T,C']
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)

            # for distillation
            if self.distill_middle and (i in self.middle_temporal_ids):
                self.middle_temporal.append(x)

            # [N*V,T,C'] -> [N,T,V,C']
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        return x

    def forward(self, x, return_rep=False):
        if self.distill_middle:
            self.middle_spatial.clear()
            self.middle_temporal.clear()

        # [N,T,V,C]
        b, f, n, c = x.shape

        # [N,T,V,C] -> [N * V, T, C']
        x = self.STE_forward(x, no=0)

        # [N * V, T, C']
        x = self.TTE_foward(x, no=0)

        # now x shape is (b n) f cw
        # [N * V, T, C'] -> [N,T,V,C']
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # st = time.time()
        x = self.ST_foward(x)
        if return_rep:
            return x

        x = self.head(x)

        x = x.view(b, f, n, -1)

        return x

    def get_representation(self, x):
        return self.forward(x, return_rep=True)


################################################################
# get flops and params
def caculate_flops_and_params():
    print('caculating flops')
    from thop import profile, clever_format
    from common.arguments import parse_args
    args = parse_args()
    args.spatial_mode = 'hyper_dependent_common'

    N, T, V, C = 1, 243, 17, 2
    # wo distill
    embedding_C = 512
    depth = 8
    # distill
    # embedding_C = 128
    # depth = 6
    x = torch.rand([N, T, V, C])

    net = MixSTE2(num_frame=T, embed_dim_ratio=embedding_C, depth=depth, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                  drop_path_rate=0, graph_mode='uniform', hyper_opt=args)
    macs, params = profile(net, inputs=(x,))
    mcas, params = clever_format([macs, params])
    flops = 2 * macs
    print("FLOPs: %s" % (flops / 1e6))
    print("params: %s" % (params))


if __name__ == '__main__':
    caculate_flops_and_params()
