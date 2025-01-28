import logging
import os
import os.path as osp
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn as nn
from functools import partial


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # 当前批次的topk准确率
        self.val = 0
        # 总的所有累计数量平均topk准确率
        self.avg = 0
        # 总的所有累计准确预测的数量
        self.sum = 0
        # 总的所有累计数量
        self.count = 0

    def update(self, val, n=1):
        '''
        @param val: 当前批次下topk的准确率
        @param n: 当前批次的batch_size大小
        '''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算top-k准确率的工具函数(这个函数要当作模板记住!!下次直接拿过来用)
def accuracy(output, target, topk=(1,)):
    '''
    @param output: [N,num_classes]
    @param target: [N,1]
    return [av_topk]
    '''
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): for agiven str path, we have load pretrained weights by torch.load and get state dict into checkpoint
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model


def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model


def load_backbone(args, cfg):
    if args.version == 1:
        from lib.model.mixste_model_plus_2 import MixSTE2 as Model
    elif args.version == 2:
        from lib.model.mixste_model_plus_2_plus import MixSTE2 as Model

    receptive_field = args.number_of_frames

    num_joints = 17
    # here please attention, we follow the setting of coordination used by PoseConv3D and MotionBERT, we set (x,y,confidence).
    # therefore, the input channel should be  C=2+1
    model_backbone = Model(num_frame=receptive_field, num_joints=num_joints, in_chans=2 if cfg.no_conf else 3,
                           embed_dim_ratio=args.cs, depth=args.dep,
                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, hyper_opt=args,
                           graph_mode=args.hyper_relation_bias, hrnet=False)

    print('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')
    logging.info('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')

    return model_backbone


def load_backbone_distill(args, cfg):
    if args.version == 1:
        from lib.model.mixste_model_plus_2 import MixSTE2 as Model
    elif args.version == 2:
        from lib.model.mixste_model_plus_2_plus import MixSTE2 as Model

    receptive_field = args.number_of_frames
    num_joints = 17
    # here please attention, we follow the setting of coordination used by PoseConv3D and MotionBERT, we set (x,y,confidence).
    # therefore, the input channel should be  C=2+1
    model_backbone_stu = Model(num_frame=receptive_field, num_joints=num_joints, in_chans=2 if cfg.no_conf else 3,
                               embed_dim_ratio=args.cs, depth=args.dep,
                               num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                               hyper_opt=args,
                               graph_mode=args.hyper_relation_bias,
                               distill_middle=True, middle_no_list=args.middle_ids_s, hrnet=args.hrdet)

    model_backbone_tea = Model(num_frame=receptive_field, num_joints=num_joints, in_chans=2 if cfg.no_conf else 3,
                               embed_dim_ratio=args.t_cs, depth=args.t_dep,
                               num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0,
                               hyper_opt=args,
                               graph_mode=args.hyper_relation_bias,
                               distill_middle=True, middle_no_list=args.middle_ids_t, hrnet=args.hrdet)

    print('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')
    logging.info('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')

    return model_backbone_tea, model_backbone_stu


def load_distiller(args, tea_model):
    from lib.model.loss_distill import Pose3DKDLoss3 as Distiller
    distiller = Distiller(teacher_model=tea_model,
                          middle_layer_ids_t=args.middle_ids_t, middle_layer_ids_s=args.middle_ids_s,
                          student_channel_dims=args.s_cs, teacher_channel_dims=args.t_cs,
                          alpha_middle_kd=args.middle_kd_alpha,
                          args=args)
    return distiller


def load_backbone_vis(args, cfg):
    if args.version == 1:
        from lib.model.mixste_model_plus_2 import MixSTE2 as Model
    elif args.version == 2:
        from lib.model.mixste_model_plus_2_plus import MixSTE2 as Model

    receptive_field = args.number_of_frames
    num_joints = 17
    # here please attention, we follow the setting of coordination used by PoseConv3D and MotionBERT, we set (x,y,confidence).
    # therefore, the input channel should be  C=2+1
    model_backbone = Model(num_frame=receptive_field, num_joints=num_joints, in_chans=2 if cfg.no_conf else 3,
                           embed_dim_ratio=args.cs, depth=args.dep,
                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, hyper_opt=args,
                           graph_mode=args.hyper_relation_bias, hrnet=False, visuzlie=True)

    print('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')
    logging.info('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')

    return model_backbone


if __name__ == '__main__':
    # # 测试tensor.topk的用法
    # N=5
    # # a = torch.arange(1, 5)
    # a = torch.randn((1,N))
    # print(a)
    # # topk_v, topk_index = torch.topk(a, k=3, dim=-1, largest=True, sorted=True)
    # topk_v, topk_index = torch.topk(a, k=3, dim=-1, largest=True, sorted=False)
    # print(topk_v, topk_index)

    # 测试这里经过topk包装的函数 accuracy
    N = 8
    num_classes = 10
    fake_output = torch.randn((N, num_classes))
    fake_label = torch.arange(0, N).view(N, -1)
    res = accuracy(output=fake_output, target=fake_label, topk=(1, 3))
    print(res)
