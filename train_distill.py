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
    args.dim_feat = args.dim_rep = opts.cs
    opts.dataset = args.dataset
    opts.log = opts.l = args.log
    args.depth = opts.dep
    args.maxlen = opts.number_of_frames


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


def save_checkpoint_distiller(chk_path, distiller):
    print('Saving checkpoint to', chk_path)
    logging.info(f'Saving checkpoint to {chk_path}')

    torch.save({
        'distiller': distiller.state_dict(),
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


# default parameter here likes :
# train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)
def train_epoch(args, model_pos_stu, model_pos_tea, distiller, train_loader, losses, optimizer, has_3d, has_gt):
    '''

    :param args: from config file
    :param model_pos_stu:
    :param train_loader:
    :param losses:
    :param optimizer:
    :param has_3d:
    :param has_gt:
    :return:
    '''
    model_pos_tea.eval()
    model_pos_stu.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:, :, :, 2:])  # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1,
                                                              2]  # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        # For teacher
        with torch.no_grad():
            model_pos_tea(batch_input)
        spatial_stu_list = model_pos_stu.module.middle_spatial
        temporal_stu_list = model_pos_stu.module.middle_temporal
        spatial_tea_list = model_pos_tea.module.middle_spatial
        temporal_tea_list = model_pos_tea.module.middle_temporal

        # For student
        predicted_3d_pos = model_pos_stu(batch_input)  # (N, T, 17, 3)

        optimizer.zero_grad()
        if has_3d:
            # currently, we use this branch
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_tcl = tcl_loss(predicted_3d_pos)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_distill = distiller.forward(
                label=batch_gt, logit_S=predicted_3d_pos,
                middle_spatial_S=spatial_stu_list[:len(spatial_stu_list) - 1]
                if opts.generation else spatial_stu_list,
                middle_spatial_T=spatial_tea_list[:len(spatial_tea_list) - 1]
                if opts.generation else spatial_tea_list,
                middle_temporal_S=temporal_stu_list[:len(temporal_stu_list) - 1]
                if opts.generation else temporal_stu_list,
                middle_temporal_T=temporal_tea_list[:len(temporal_tea_list) - 1]
                if opts.generation else temporal_tea_list,
                last_temporal_S=temporal_stu_list[-1] if opts.generation else None,
                last_temporal_T=temporal_tea_list[-1] if opts.generation else None
            )
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)

            # last 4 loss are not used in current train stage
            loss_total = loss_3d_pos + \
                         args.lambda_scale * loss_3d_scale + \
                         args.lambda_tcl * loss_3d_tcl + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         loss_distill + \
                         args.lambda_lv * loss_lv + \
                         args.lambda_lg * loss_lg + \
                         args.lambda_a * loss_a + \
                         args.lambda_av * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_tcl'].update(loss_3d_tcl.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['distill'].update(loss_distill.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_distill = distiller.forward(
                label=batch_gt, logit_S=predicted_3d_pos,
                middle_spatial_S=spatial_stu_list[:len(spatial_stu_list) - 1]
                if args.generation else spatial_stu_list,
                middle_spatial_T=spatial_tea_list[:len(spatial_tea_list) - 1]
                if args.generation else spatial_tea_list,
                middle_temporal_S=temporal_stu_list[:len(temporal_stu_list) - 1]
                if args.generation else temporal_stu_list,
                middle_temporal_T=temporal_tea_list[:len(temporal_tea_list) - 1]
                if args.generation else temporal_tea_list,
                last_temporal_S=temporal_stu_list[-1] if args.generation else None,
                last_temporal_T=temporal_tea_list[-1] if args.generation else None
            )
            loss_total = loss_2d_proj + loss_distill
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['distill'].update(loss_distill.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()

        if args.quickdebug:
            break


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

    try:
        os.makedirs(opts.checkpoint, exist_ok=True)
        # make soft link for log and checkpoint
        os.symlink(opts.checkpoint, args.log)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print('Loading dataset...')
    logging.info('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)

    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                dt_root='data/motion3d/H36M-SH', dt_file=args.dt_file)
    min_loss = math.inf

    tea_model_backbone, stu_model_backbone = load_backbone_distill(opts, args)
    distiller = load_distiller(opts, tea_model_backbone)
    tea_model_params = 0
    stu_model_params = 0
    for parameter in tea_model_backbone.parameters():
        tea_model_params = tea_model_params + parameter.numel()
    print('INFO: Trainable parameter count:', tea_model_params / 1000000, 'Million')
    logging.info(f'INFO: Trainable parameter count: {tea_model_params / 1000000} Million')

    for parameter in stu_model_backbone.parameters():
        stu_model_params = stu_model_params + parameter.numel()
    print('INFO: Trainable parameter count:', stu_model_params / 1000000, 'Million')
    logging.info(f'INFO: Trainable parameter count: {stu_model_params / 1000000} Million')

    if torch.cuda.is_available():
        tea_model_backbone = nn.DataParallel(tea_model_backbone).cuda()
        stu_model_backbone = nn.DataParallel(stu_model_backbone).cuda()
        distiller = nn.DataParallel(distiller).cuda()

    # for teacher model
    tea_chk_filename = os.path.join(opts.tea_ckpt)
    print('Loading teacher checkpoint', tea_chk_filename)
    logging.info(f'Loading teacher checkpoint {tea_chk_filename}')
    checkpoint = torch.load(tea_chk_filename, map_location=lambda storage, loc: storage)
    tea_model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

    model_pos_tea = tea_model_backbone
    model_pos_stu = stu_model_backbone

    # for student model and distiller resume
    if opts.resume:
        chk_filename = os.path.join(opts.stu_ckpt)
        print('Loading student checkpoint', chk_filename)
        logging.info(f'Loading student checkpoint {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        stu_model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos_stu = stu_model_backbone
        print('Loading distiller...')
        logging.info('Loading distiller...')
        distiller_path = osp.join(opts.distiller_ckpt)
        distiller_ckpt = torch.load(distiller_path, map_location=lambda storage, loc: storage)
        distiller.load_state_dict(distiller_ckpt['distiller'], strict=False)

    if args.partial_train:
        model_pos_stu = partial_train_layers(model_pos_stu, args.partial_train)

    lr = args.learning_rate
    # Optimizer: different distill type determines optimizer setting
    if hasattr(distiller.module, 'temporal_align') and hasattr(distiller.module, 'spatial_align'):
        optimizer = optim.AdamW(list(model_pos_stu.parameters()) +
                                list(distiller.parameters()),
                                lr=lr,
                                weight_decay=args.weight_decay)
    elif hasattr(distiller.module, 'temporal_align'):
        optimizer = optim.AdamW(list(model_pos_stu.parameters()) + list(distiller.module.temporal_align.parameters()),
                                lr=lr,
                                weight_decay=args.weight_decay)
    elif hasattr(distiller.module, 'spatial_align'):
        optimizer = optim.AdamW(list(model_pos_stu.parameters()) + list(distiller.module.spatial_align.parameters()),
                                lr=lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model_pos_stu.parameters(),
                                lr=lr,
                                weight_decay=args.weight_decay)

    lr_decay = args.lr_decay
    st = 0
    if args.train_2d:
        print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d),
                                                               len(instav_loader_2d) + len(posetrack_loader_2d)))
        logging.info('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d),
                                                                      len(instav_loader_2d) + len(
                                                                          posetrack_loader_2d)))
    else:
        print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        logging.info('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
    if opts.resume:
        st = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(
                'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            logging.info(
                'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        lr = checkpoint['lr']
        if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
            min_loss = checkpoint['min_loss']

    args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
    if args.mask or args.noise:
        # data autumentation
        args.aug = Augmenter2D(args)

    # Training
    for epoch in range(st, args.epochs):
        print('Training epoch %d.' % (epoch + 1))
        logging.info('Training epoch %d.' % (epoch + 1))
        start_time = time()
        losses = {}
        losses['3d_pos'] = AverageMeter()
        losses['3d_scale'] = AverageMeter()
        losses['3d_tcl'] = AverageMeter()
        losses['distill'] = AverageMeter()
        losses['2d_proj'] = AverageMeter()
        losses['lg'] = AverageMeter()
        losses['lv'] = AverageMeter()
        losses['total'] = AverageMeter()
        losses['3d_velocity'] = AverageMeter()
        losses['angle'] = AverageMeter()
        losses['angle_velocity'] = AverageMeter()
        N = 0

        # Curriculum Learning 表示从易到难的学习,默认的训练阶段不进入
        if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
            train_epoch(args, model_pos_stu, model_pos_tea, distiller, posetrack_loader_2d, losses, optimizer,
                        has_3d=False, has_gt=True)
            train_epoch(args, model_pos_stu, model_pos_tea, distiller, instav_loader_2d, losses, optimizer,
                        has_3d=False, has_gt=False)
        # currently, we enter this branch
        train_epoch(args, model_pos_stu, model_pos_tea, distiller, train_loader_3d, losses, optimizer, has_3d=True,
                    has_gt=True)
        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f distill_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                losses['distill'].avg))
            logging.info('[%d] time %.2f lr %f 3d_train %f distill_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                losses['distill'].avg))
        else:
            e1, e2, results_all = evaluate(args, model_pos_stu, test_loader, datareader)
            print('[%d] time %.2f lr %f 3d_train %f distill_train %f e1 %f e2 %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                losses['distill'].avg,
                e1, e2))
            logging.info('[%d] time %.2f lr %f 3d_train %f distill_train %f e1 %f e2 %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                losses['distill'].avg,
                e1, e2))

            print(f'current best e1 is {min_loss}')
            logging.info(f'current best e1 is {min_loss}')

            train_writer.add_scalar('Error P1', e1, epoch + 1)
            train_writer.add_scalar('Error P2', e2, epoch + 1)
            train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
            train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
            train_writer.add_scalar('loss_3d_tcl', losses['3d_tcl'].avg, epoch + 1)
            train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
            train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
            train_writer.add_scalar('loss_distill', losses['distill'].avg, epoch + 1)
            train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
            train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
            train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
            train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
            train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # Save checkpoints settings
        chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
        chk_distiller_path = os.path.join(opts.checkpoint, 'epoch_{}_distiller.bin'.format(epoch))
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_distiller_path_latest = os.path.join(opts.checkpoint, 'latest_epoch_distiller.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
        chk_path_distiller_best = os.path.join(opts.checkpoint, 'best_epoch_distiller.bin'.format(epoch))

        # There are three situation need to save
        # save latest checkpoint
        save_checkpoint(chk_path_latest, epoch + 1, lr, optimizer, model_pos_stu, min_loss)
        save_checkpoint_distiller(chk_distiller_path_latest, distiller)

        # save frequency checkpoint
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_checkpoint(chk_path, epoch + 1, lr, optimizer, model_pos_stu, min_loss)
            save_checkpoint_distiller(chk_distiller_path, distiller)

        # save best checkpoint
        if e1 < min_loss:
            min_loss = e1
            save_checkpoint(chk_path_best, epoch + 1, lr, optimizer, model_pos_stu, min_loss)
            save_checkpoint_distiller(chk_path_distiller_best, distiller)


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    args = get_config(opts.config)
    reload_certain_task_attribute(opts, args)
    register_log_info(opts)
    set_random_seed(opts.seed)
    train_with_config(args, opts)
