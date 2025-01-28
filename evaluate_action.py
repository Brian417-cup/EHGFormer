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
                        time.strftime('%m%d_%H%M_%S_') + f'{opts.dataset}_evaluate.log')
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
    opts.dataset = args.dataset
    opts.dep = args.depth
    opts.cs = args.dim_rep
    opts.log = opts.l = args.log
    opts.number_of_frames = args.maxlen


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


def evaluate_with_config(args, opts):
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

    # load evaluate model definition
    model_backbone = load_backbone(opts)

    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes,
                      dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,
                      num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')
    logging.info(f'INFO: Trainable parameter count: {model_params / 1000000} Million')

    print(f'Loading test set from {args.dataset}...')
    logging.info(f'Loading test set from {args.dataset}...')

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset

    # 评估阶段的数据加载器
    ntu_val = NTURGBD(data_path=data_path, data_split=args.data_split + '_val', n_frames=args.clip_len,
                      random_move=False, scale_range=args.scale_range_test)

    test_loader = DataLoader(ntu_val, **testloader_params)

    # resume checkpoint from disk, it's different from partial resume. It resume for complete network
    # resume for checkpoint weight

    assert opts.evaluate != '', "In evaluation stage, corresponding checkpoint path shouln't be None !!"
    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)
    logging.info(f'Loading checkpoint {chk_filename}')

    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    load_pretrained_weights(model, checkpoint['model'])

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    # 评估模式

    test_loss, test_top1, test_top5 = validate(test_loader, model, criterion)

    print('Loss {loss:.4f} \t'
          'Acc@1 {top1:.3f} \t'
          'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
    logging.info('Loss {loss:.4f} \t'
                 'Acc@1 {top1:.3f} \t'
                 'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    args = get_config(opts.config)
    reload_certain_task_attribute(opts, args)
    register_log_info(args)
    set_random_seed(opts.seed)
    evaluate_with_config(args, opts)
