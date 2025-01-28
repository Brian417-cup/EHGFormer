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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from lib.utils.load_parameter import parse_args


#################################################################
# register log
def register_log_info(opts):
    '''
    :param opts: from cmd
    :return:
    '''
    import time
    from lib.utils.my_logger import init_logger
    os.makedirs(opts.log, exist_ok=True)

    log_path = osp.join(opts.log,
                        time.strftime('%m%d_%H%M_%S_') + f'{opts.dataset}.log')
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
    if args.finetune:
        assert args.backbone_ckpt is not None, 'Pretrained backbone checkpoint path should not be None!!'
    opts.number_of_frames = args.maxlen
    opts.dataset = args.dataset
    opts.dep = args.depth
    opts.cs = args.dim_rep
    opts.log = opts.l = args.log


#################################################################

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)  # (N, num_classes)
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    idx, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    idx, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):
    '''

    :param args: from config file *.yaml
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
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "tensorboard_logs"))
    # 加载网络结构
    model_backbone = load_backbone(opts)

    # here is finetune stargegy: Please attention, finetune here only means finetune backbone rather than total network(with calssification head) !!
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            backbone_chk_filename = os.path.join(args.backbone_ckpt)

            print('Loading backbone', backbone_chk_filename)
            logging.info(f'Loading backbone {backbone_chk_filename}')

            load_ckpt = torch.load(backbone_chk_filename, map_location=lambda storage, loc: storage)
            if 'model_pos' in load_ckpt.keys():
                checkpoint = load_ckpt['model_pos']
            elif 'model' in load_ckpt.keys():
                checkpoint = load_ckpt['model']
            else:
                assert NotImplementedError

            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes,
                      dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,
                      num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')
    logging.info(f'INFO: Trainable parameter count: {model_params / 1000000} Million')

    print('Loading dataset...')
    logging.info('Loading dataset...')

    # If ntu120, it need larger memory
    if args.dataset == 'ntu120_hrnet':
        trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': False,
            # 'prefetch_factor': 4,
            # 'persistent_workers': False
        }
        testloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': False,
            # 'prefetch_factor': 4,
            # 'persistent_workers': False
        }
    else:
        trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
        testloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
    data_path = 'data/action/%s.pkl' % args.dataset
    # 训练阶段的数据加载器
    ntu_train = NTURGBD(data_path=data_path, data_split=args.data_split + '_train', n_frames=args.clip_len,
                        random_move=args.random_move, scale_range=args.scale_range_train)
    # 评估阶段的数据加载器
    ntu_val = NTURGBD(data_path=data_path, data_split=args.data_split + '_val', n_frames=args.clip_len,
                      random_move=False, scale_range=args.scale_range_test)

    train_loader = DataLoader(ntu_train, **trainloader_params)
    test_loader = DataLoader(ntu_val, **testloader_params)

    # resume checkpoint from disk, it's different from partial resume. It resume for complete network
    # resume for checkpoint weight
    if opts.resume or opts.evaluate:
        # evaluation has been removed from current script.
        if opts.evaluate:
            assert NotImplementedError, 'Please run special evaluation script about evaluate_action.py, coresponding commands can be from README.md!!'
        # for resume checkpoint and continue to train
        else:
            backbone_chk_filename = opts.resume
            print('Loading checkpoint', backbone_chk_filename)
            logging.info(f'Loading checkpoint {backbone_chk_filename}')

            checkpoint = torch.load(backbone_chk_filename, map_location=lambda storage, loc: storage)
            load_pretrained_weights(model, checkpoint['model'])

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    # train stage
    # optimizer and lr
    if not opts.evaluate:
        optimizer = optim.AdamW(
            [{"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
             {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
             ], lr=args.lr_backbone,
            weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0

        print('INFO: Training on {} batches'.format(len(train_loader)))
        logging.info('INFO: Training on {} batches'.format(len(train_loader)))

        # resume for epoch, learning rate and best_acc for last resume epoch
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
                logging.info(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            if 'lr' in checkpoint and checkpoint['lr'] is not None:
                lr = checkpoint['lr']
                print(f'Last lr is {lr}')
                logging.info(f'Last lr is {lr}')

            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
                print(f'Last best acc is {best_acc}')
                logging.info(f'Last best acc is {best_acc}')

        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % (epoch + 1))
            logging.info('Training epoch %d.' % (epoch + 1))

            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):  # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model.forward(batch_input)  # output shape: (N, num_classes)

                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()

                if args.quickdebug:
                    break

            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses_train, top1=top1))
                logging.info('Train: [{0}][{1}/{2}]\t'
                             'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses_train, top1=top1))

                sys.stdout.flush()

            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion)

            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)

            # scheduler for each epoch
            scheduler.step()

            print(
                f'current epoch {epoch + 1} lr is {scheduler.get_last_lr()}  top1 acc is {test_top1}, best top1 acc is {best_acc}.')
            logging.info(
                f'current epoch {epoch + 1} lr is {scheduler.get_last_lr()}  top1 acc is {test_top1}, best top1 acc is {best_acc}.')

            # Save cur epoch checkpoint if need for certain save frequency.
            latest_chk_path = os.path.join(opts.checkpoint, f'latest.bin')
            chk_path = os.path.join(opts.checkpoint, f'epoch_{epoch + 1}.bin')

            # save latest epoch bin
            print(f'Saving latest checkpoint to {latest_chk_path}')
            logging.info(f'Saving latest checkpoint to {latest_chk_path}')
            torch.save({
                'epoch': epoch + 1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc': best_acc
            }, latest_chk_path)

            # 这里作者做了更加详细的相关参数更新，将epoch,lr,optimizer和best_acc的参数量都更新进去了
            if epoch % args.ckpt_save_frequency == 0:
                print('Saving checkpoint to', chk_path)
                logging.info(f'Saving checkpoint to {chk_path}')

                torch.save({
                    'epoch': epoch + 1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc': best_acc
                }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1

                print(f"save best checkpoint")
                logging.info("save best checkpoint")
                print(f'current best accuracy is updated to {best_acc}')
                logging.info(f'current best accuracy is updated to {best_acc}')

                torch.save({
                    'epoch': epoch + 1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc': best_acc
                }, best_chk_path)

            if args.quickdebug:
                break

    # evaluation stage
    if opts.evaluate:
        assert NotImplementedError, 'Please run special evaluation script about evaluate_action.py, coresponding commands can be from README.md!!'


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    args = get_config(opts.config)
    reload_certain_task_attribute(opts, args)
    register_log_info(opts)
    set_random_seed(opts.seed)
    train_with_config(args, opts)
