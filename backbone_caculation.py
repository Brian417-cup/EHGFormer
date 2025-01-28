# run this command:
# python backbone_caculation.py --version 2 --config configs/action/train_NTU60_xsub.yaml --hyper_cfg hyper_config/method2/manual_joint_label3.yaml --hyper_relation_bias uniform
#
import logging
import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
from torch.utils import tensorboard as tensorboardX
from tqdm import tqdm
import random

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from lib.utils.load_parameter import parse_args


#################################################################
def reload_certain_task_attribute(opts):
    opts.dataset = 'ntu'
    opts.version = 2
    # opts.config = r'configs/action/train_NTU60_xsub.yaml'
    # opts.hyper_cfg = r'hyper_config/method2/manual_joint_label3.yaml'
    # opts.hyper_relation_bias = 'uniform'
    # opts.spatial_mode='hyper_dependent_common'


#################################################################
def train_with_config(args, opts):
    '''

    :param args: from config file *.yaml
    :param opts: from cmd
    :return:
    '''
    print('#####################################')
    print('config attribute')
    print(args)
    print('#####################################')
    print('cmd attribute')
    print(opts)

    # 加载网络结构
    model_backbone = load_backbone(opts)

    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    torch.save(model_backbone.state_dict(),'init_pose3d.bin')

    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes,
                      dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,
                      num_joints=args.num_joints)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    torch.save(model.state_dict(),'last.bin')


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    reload_certain_task_attribute(opts)
    args = get_config(opts.config)
    train_with_config(args, opts)
