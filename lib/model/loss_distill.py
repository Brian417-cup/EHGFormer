# This loss is for distill
import os.path as osp
import sys
import time

import numpy as np

sys.path.append(osp.join(osp.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import logging
from einops import rearrange


# Common KD for logit in the last classification head
class CommonKDLoss(nn.Module):
    def __init__(self, alpha=0.5, temprature=1.0, teacher_model: nn.Module = None, base_criterion=None):
        assert teacher_model is not None, "Teacher model for knowledge distillation shouln't be None"
        super(CommonKDLoss, self).__init__()

        self.base_criterion = base_criterion if base_criterion is not None else self._inter_base_criterion

        self.temperature = temprature
        self.alpha = alpha
        self.teacher_model = teacher_model.eval()
        if torch.cuda.is_available():
            self.teacher_model = self.teacher_model.cuda()

    def forward(self, inputs: torch.Tensor, student_logits: torch.Tensor, label: torch.Tensor):
        '''

        Args:
            inputs:
            student_logits: [N,num_class]
            label: [N]
        Returns:

        '''
        with torch.no_grad():
            # [N, num_class]
            teacher_logits = self.teacher_model(inputs)
        S_i = F.softmax(student_logits / self.temperature, dim=1)
        T_i = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = -self.alpha * (self.temperature ** 2) * (T_i * torch.log(S_i)).sum(dim=1).mean()
        base_loss = self.base_criterion(student_logits, label)
        return (1 - self.alpha) * base_loss + kd_loss

    def _inter_base_criterion(self):
        return nn.CrossEntropyLoss()


# Modified by VitKDLoss from https://github.com/yzd-v/cls_KD/blob/1e5d48f36957ee460d39d7ebc592c6cc7f8f3d66/mmcls/models/dis_losses/vitkd.py#L8
# its main function is for middle layer distillation
# other reference comes from
# https://github.com/ilovepose/fast-human-pose-estimation.pytorch/blob/c4722f8593305e2b8c37601774c5d5c3cd668a01/tools/fpd_train.py#L31
# https://github.com/huawei-noah/Efficient-Computing/blob/master/Distillation/ManifoldKD/losses.py
# Used for Pose3D in temporal dimension
# some question about mimicking
class Pose3DKDLoss3(nn.Module):
    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 teacher_model: nn.Module = None,
                 middle_layer_ids_t: List[int] = [],
                 middle_layer_ids_s: List[int] = [],
                 student_channel_dims=512,
                 teacher_channel_dims=512,
                 alpha_middle_kd=0.003,
                 args=None
                 ):
        assert teacher_model is not None, "Teacher model for knowledge distillation shouln't be None!!"
        # assert base_criterion is not None, "Base loss function must be not None!!"
        assert len(middle_layer_ids_s) == len(middle_layer_ids_t), \
            'Count of middle layers between teacher and student must be same!!'
        super(Pose3DKDLoss3, self).__init__()

        self.args = args

        # Common KD (Not used currently)
        # self.alpha_kd = alpha_kd

        # VitKD (mimicking part for middle layers)
        self.middle_layer_ids_s = middle_layer_ids_s
        self.middle_layer_ids_t = middle_layer_ids_t
        self.alpha_middle_kd = alpha_middle_kd

        if len(self.middle_layer_ids_t) > 0:
            if student_channel_dims != teacher_channel_dims:
                self.spatial_align = nn.ModuleList([
                    nn.Linear(student_channel_dims, teacher_channel_dims, bias=True)
                    for i in range(len(middle_layer_ids_s))])
                self.temporal_align = nn.ModuleList([
                    nn.Linear(student_channel_dims, teacher_channel_dims, bias=True)
                    for i in range(len(middle_layer_ids_s))])
            else:
                self.spatial_align = None
                self.temporal_align = None

        # self.teacher_model = teacher_model
        # self.teacher_model.eval()
        # make model parallel
        # if torch.cuda.is_available():
        #     self.teacher_model = nn.DataParallel(self.teacher_model).cuda()

        # Generation part for last layer
        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_channel_dims))
        self.generation_align = nn.Linear(student_channel_dims, teacher_channel_dims, bias=True)
        self.generation_compare = nn.Sequential(
            nn.Conv1d(teacher_channel_dims, teacher_channel_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(teacher_channel_dims, teacher_channel_dims, kernel_size=3, padding=1)
        )
        self.generation_tds = self.args.generation_tds

    def forward(self,
                # x,
                label,
                logit_S,
                middle_spatial_S,
                middle_spatial_T,
                middle_temporal_S,
                middle_temporal_T,
                last_temporal_S=None,
                last_temporal_T=None
                ):
        """Forward function.
        Args:
            x: [N,T,V,C=2]
            label: [N,T,V,C=3]
            logit_S: [N,T,V,C=3]
            preds_S(List): [B*2*N*D], student's feature map
            preds_T(List): [B*2*N*D], teacher's feature map
        """
        assert label.shape == logit_S.shape, 'Output shape for student model with label should be same!!'

        # Common KD(w/o middle distillation)
        if len(self.middle_layer_ids_t) == 0:
            assert NotImplementedError
        # w middle distillation
        else:
            # Mimicking
            spatial_middle_criterion = nn.MSELoss(reduction='sum')
            temporal_middle_criterion = nn.MSELoss(reduction='sum')

            spatial_losses = 0
            temporal_losses = 0
            for i, (id_s, id_t) in enumerate(zip(self.middle_layer_ids_s, self.middle_layer_ids_t)):
                # exclude generation part
                if id_s == self.args.s_dep - 1 or id_t == self.args.t_dep - 1:
                    continue

                spatial_f_s = middle_spatial_S[i]
                temporal_f_s = middle_temporal_S[i]

                batch_spatial = spatial_f_s.shape[0]
                batch_temporal = temporal_f_s.shape[0]

                with torch.no_grad():
                    spatial_f_t = middle_spatial_T[i]
                    temporal_f_t = middle_temporal_T[i]

                if self.spatial_align is not None:
                    spatial_f_s = self.spatial_align[i](spatial_f_s)

                if self.temporal_align is not None:
                    temporal_f_s = self.temporal_align[i](temporal_f_s)

                spatial_losses += spatial_middle_criterion(spatial_f_s, spatial_f_t) / batch_spatial
                temporal_losses += temporal_middle_criterion(temporal_f_s, temporal_f_t) / batch_temporal

            # Generation (only for last layer temporal feature)
            loss_gen = 0
            if last_temporal_S is not None and last_temporal_T is not None:
                loss_generation_mse = nn.MSELoss(reduction='sum')
                if self.generation_align is not None:
                    last_temporal_S = self.generation_align(last_temporal_S)

                N, T, C = last_temporal_S.shape
                last_temporal_S_keep, mask_binary, ids_mask, ids_keep = self.unifrom_masking(last_temporal_S,
                                                                                             tds=self.generation_tds)
                last_temporal_S[:, ids_mask, :] = self.mask_token.repeat([N, 1, 1])
                last_temporal_S = self.generation_compare(rearrange(last_temporal_S, 'n t c -> n c t'))
                last_temporal_S = rearrange(last_temporal_S, 'n c t -> n t c')
                loss_gen = loss_generation_mse(torch.mul(last_temporal_S, mask_binary),
                                               torch.mul(last_temporal_T, mask_binary))
                loss_gen = loss_gen / (N * (1 - 1 / self.generation_tds))

            middle_kd = self.alpha_middle_kd * spatial_losses + self.alpha_middle_kd * temporal_losses + self.alpha_middle_kd * loss_gen
            # logging.info(f'middle loss is {middle_kd},base loss is {base_loss}')
            return middle_kd

    def unifrom_masking(self, x, tds=2):
        '''
        Perform downsample on temporal masking.
        Here, we have a assumption that the first frame is kept.
        Args:
            x: [N, L, D], sequence
            tds: uniform downsample mask

        Returns:
            x_keep: unmasked token [N,L,D]
            mask_binary: [N,L,D] 0 is keep, 1 is masked
            ids_mask: [L-L//tds] in the init sequence, index list of unmasked for temporal dimension
            ids_keep: [L//tds] in the init sequence, index list of masked for temporal dimension
        '''

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L // tds)

        # generate keep index and mask index
        ############################################
        all_temporal_idxs = torch.arange(0, L)
        ids_keep = torch.arange(0, L, tds)
        ids_mask = [item for item in all_temporal_idxs.numpy().tolist() if item not in ids_keep.numpy().tolist()]
        ids_mask = torch.tensor(ids_mask, device=x.device)
        ids_keep = ids_keep.to(x.device)
        ############################################

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(0).unsqueeze(-1).repeat(N, 1, D))

        # generate mask: 0 is keep, 1 is remove
        mask_binary = torch.ones([N, L], device=x.device)
        mask_binary[:, ids_keep] = 0
        mask_binary = torch.unsqueeze(mask_binary, dim=-1)

        return x_keep, mask_binary, ids_mask, ids_keep
