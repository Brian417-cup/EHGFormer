import os
import numpy as np
import argparse
import errno
import math
import pickle
from torch.utils import tensorboard as tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
# from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_h36m_tds import DataReaderH36M
from lib.model.loss import *
from lib.utils.load_parameter import parse_args


# register log
def register_log_info(args):
    '''
    :param args: from cmd
    :return:
    '''
    import time
    from lib.utils.my_logger import init_logger
    os.makedirs(args.log, exist_ok=True)

    log_path = osp.join(args.log,
                        time.strftime('%m%d_%H%M_%S_') + f'{args.dataset}.log')
    logger = init_logger(log_path, 'my_logger')
    return logger


# set fix seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reload_certain_task_attribute(opts, args):
    '''
    Reload som key parameter that has been defined in total config and update into arguments.
    :param opts: from cmd
    :param args: from config
    :return:
    '''
    assert args.dim_feat == args.dim_rep, 'Channel of regression head and encoders should be same!!'
    opts.dataset = args.dataset
    opts.cs = args.dim_rep
    opts.log = opts.l = args.log
    opts.dep = args.depth
    opts.number_of_frames = args.maxlen


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    logging.info(f'Saving checkpoint to {chk_path}')

    torch.save({
        'epoch': epoch,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss
    }, chk_path)


def evaluate(args, model_pos, test_loader, datareader):
    '''

    :param args:
    :param model_pos:
    :param test_loader:
    :param datareader:
    :return:
    '''
    print('INFO: Testing')
    logging.info('INFO: Testing')

    results_all = []
    model_pos.eval()
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                batch_gt[:, 0, 0, 2] = 0

            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

            # if args.quickdebug:
            #     break

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    # actions = np.array(datareader.dt_dataset['test']['action'])
    # factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    # gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    # sources = np.array(datareader.dt_dataset['test']['source'])

    # use tds
    actions = np.array(datareader.dt_dataset['test']['action'][::args.sample_stride])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'][::args.sample_stride])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'][::args.sample_stride])
    sources = np.array(datareader.dt_dataset['test']['source'][::args.sample_stride])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]

    # This assert has problem now
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]

        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    logging.info(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    logging.info(f'Protocol #1 Error (MPJPE): {e1} mm')
    print(f'Protocol #2 Error (P-MPJPE): {e2} mm')
    logging.info(f'Protocol #2 Error (P-MPJPE): {e2} mm')
    print('----------')
    logging.info('----------')
    return e1, e2, results_all

def train_with_config(args, opts):
    '''

    :param args: from config file
    :param opts: from cmd
    :return:
    '''
    print('#####################################')
    logging.info('#####################################')
    print('input command')
    logging.info('input command')
    print(f"python {' '.join(sys.argv)}")
    logging.info(f"python {' '.join(sys.argv)}")

    print('#####################################')
    logging.info('#####################################')
    print('config attribute')
    logging.info('config attribute')
    print(args)
    logging.info(args)

    print('#####################################')
    logging.info('#####################################')
    print('cmd attribute')
    logging.info('cmd attribute')
    print(opts)
    logging.info(opts)
    print('#####################################')
    logging.info('#####################################')

    print('Loading dataset...')
    logging.info('Loading dataset...')

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(test_dataset, **testloader_params)

    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                dt_root='data/motion3d/H36M-SH', dt_file=args.dt_file)
    # model_backbone = load_backbone(args)
    model_backbone = load_backbone(opts)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')
    logging.info(f'INFO: Trainable parameter count: {model_params / 1000000} Million')

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    assert opts.evaluate != '', "In evaluation stage, corresponding checkpoint path shouln't be None !!"
    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)
    logging.info(f'Loading checkpoint {chk_filename}')
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone

    e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    args = get_config(opts.config)
    reload_certain_task_attribute(opts, args)
    register_log_info(opts)
    set_random_seed(opts.seed)
    train_with_config(args, opts)
