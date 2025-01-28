import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '../../..'))

import torch
from torch import nn
import torch.nn.functional as F
from lib.model.hyper_module_method2_plus.trans_spatial import Block as CommonSpatial_Block
from lib.model.hyper_module_method2_plus.trans_hyper_frame_seperation import HyperAttnBlock as HyperSpatial_Block


class CommonAndHyperSpatialTrans(nn.Module):
    def __init__(self, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, attn_drop_rate, drop_rate,
                 norm_layer, A, graph_mode, num_joints, hyper_opt, visuzlie=False):
        '''
        # Plase note : in this sequence, the hyper spatial is only supported for frame seperation spatial, which is the new method for the method2

        :param embed_dim_ratio (int): embedding dimension ratio
        :param num_heads (int): number of attention heads
        :param mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        :param attn_drop_rate (float): attention dropout rate
        :param drop_rate (float): dropout rate
        :param norm_layer: (nn.Module): normalization layer
        :param A: npt.NDArray[np.float32] the learnable matrix
        :param num_joints (int, tuple): joints number
        :param hyper_opt: arguments from cmd, especially for the hyper parameter
        :param visuzlie: if need visualize such as attention map,it should be True, default mode is False
        '''
        super().__init__()

        self.opt = hyper_opt

        self.common_attn_blk = CommonSpatial_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rate, norm_layer=norm_layer,
            A=A, graph_mode=graph_mode,
            mode=hyper_opt.spatial_mode, vis=visuzlie)

        if hyper_opt.j2e:
            self.hyper_attn_blk = HyperSpatial_Block(
                in_channels=embed_dim_ratio, out_channels=embed_dim_ratio, num_joint=num_joints,
                num_heads=num_heads, mlp_hidden_dim=int(mlp_ratio * embed_dim_ratio), qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rate, norm_layer=norm_layer,
                A=A, graph_mode=graph_mode,
                joint_label=hyper_opt.joint_label,
                spatial_mode='v', hyper_head=hyper_opt.hyper_head, multi_merge_mode=hyper_opt.hyper_multi_merge_mode,
                visualize=visuzlie)
        else:
            self.hyper_attn_blk = nn.Identity()

    def forward(self, x, T=243):
        '''

        :param x: [N*T,V,C]
        :param T:
        :return: [N*T,V,C] , [V] , [V,E<=V]
        '''
        if self.opt.j2e:
            x = self.common_attn_blk(x, T=T)
            x, joint_label, hyper_matrix = self.hyper_attn_blk(x, T=T)
            # x = x + y
            return x, joint_label, hyper_matrix
        else:
            x = self.common_attn_blk(x, T=T)
            return x
