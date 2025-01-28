import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '../../..'))

from functools import partial
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.hyper_module_method2_plus.drop_path import DropPath
from lib.model.hyper_module_method2_plus.mlp import Mlp
from lib.model.hyper_module_method2_plus import abladation_study


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, A=None,
                 graph_mode='none',
                 vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # nn.init.kaiming_normal_(self.qkv.weight)
        # torch.nn.init.xavier_uniform_(self.qkv.weight)
        # torch.nn.init.zeros_(self.qkv.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # nn.init.kaiming_normal_(self.proj.weight)
        # torch.nn.init.xavier_uniform_(self.proj.weight)
        # torch.nn.init.zeros_(self.proj.bias)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb

        ############################################################
        # skeleton adjacency matrix
        A = A.sum(0)

        if graph_mode == 'none':
            self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0),
                                      requires_grad=True)
        elif graph_mode == 'uniform':
            self.outer = nn.Parameter(torch.stack([torch.from_numpy(A).float() for _ in range(num_heads)], dim=0),
                                      requires_grad=True)

        # We currently change one -> zero here -> one
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True) if abladation_study.j2j_adjacency \
            else nn.Parameter(torch.zeros(1), requires_grad=False)
        ###########################################################

        # For visualize stage, this value is not None
        self.vis = vis
        self.attn_map = None

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k + self.alpha * self.outer) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1) + self.alpha * self.outer) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.vis:
            self.attn_map = attn.clone()

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0,
                 A=None, graph_mode='none',
                 vis=False, mode='spatial'):
        '''

        Args:
            dim:
            num_heads:
            mlp_ratio:
            attention:
            qkv_bias:
            qk_scale:
            drop:
            attn_drop:
            drop_path:
            act_layer:
            norm_layer:
            comb:
            changedim:
            currentdim:
            depth:
            vis:
        '''
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, A=A, graph_mode=graph_mode, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 针对输出通道的change策略(在1*1conv上的变化)
        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.mode = mode
        self.vis = vis

    def forward(self, x, vis=False, T=243):
        '''

        :param x: [N*T,V,C]
        :param vis:
        :return: [N*T,V,C]
        '''
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 下面这段代码在实际MixSTE结构中这边是都没有进入的，因此选择注释
        # if self.changedim and self.currentdim < self.depth // 2:
        #     print('进入该分支 reduction')
        #     x = rearrange(x, 'b t c -> b c t')
        #     x = self.reduction(x)
        #     x = rearrange(x, 'b c t -> b t c')
        # elif self.changedim and self.depth > self.currentdim > self.depth // 2:
        #     print('进入该分支 improve')
        #     x = rearrange(x, 'b t c -> b c t')
        #     x = self.improve(x)
        #     x = rearrange(x, 'b c t -> b t c')

        # print(f'进入了,{self.mode} shape', x.shape)
        return x
