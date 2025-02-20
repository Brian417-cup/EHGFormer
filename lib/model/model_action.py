import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=1024):
        super(ActionHeadClassification, self).__init__()

        hidden_dim = max(dim_rep * 2, 1024)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)  # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)  # mean in temporal dimension, result is (N,M,J,C)
        feat = feat.reshape(N, M, -1)  # (N, M, J*C)
        feat = feat.mean(dim=1)  # mean in person dimension, result is (N,J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)  # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)  # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat


class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0., version='class', hidden_dim=2048,
                 num_joints=17):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version == 'class':
            # This branch is for final classification
            self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes,
                                                 num_joints=num_joints)
        elif version == 'embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim,
                                        num_joints=num_joints)
        else:
            raise Exception('Version Error.')

    def forward(self, x):
        '''
            Input: (N, M x T x 17 x 3(here 3=2+1,first 2 is (x,y) and 3th is frame_cof))
        '''
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        # feature is in [N*M,T,V,C]
        feat = self.backbone.get_representation(x)

        # for channel, we choose first 2 channel C=2 (old version)
        # feat = self.backbone.get_representation(x[..., :2])

        # [N*M,T,V,C] -> [N,M,T,V,C]
        feat = feat.reshape([N, M, T, self.feat_J, -1])

        out = self.head(feat)
        return out
