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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False,
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

        # In order to visualize attention map and other to demonstrate temporal attention
        self.vis = vis
        self.attn_map = None

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
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
                 depth=0, mode='temporal',
                 visualize=False):
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
            visualize: If need visualize, it should save the temporal attention map
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
            comb=comb, vis=visualize)
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
        self.visisualize = visualize
        self.mode = mode

    def forward(self, x, vis=False):
        # [N,T,V*C]
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        # print(f'{self.mode} shape', x.shape)
        return x
