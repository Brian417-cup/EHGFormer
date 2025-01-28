import os
import os.path as osp
import torch


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='', help='Init raw checkpoint for pytorch.')
    return parser.parse_args()


args = get_args()
root_path = osp.join('..')

ckpt_path = osp.join(root_path, args.c)
ckpt = torch.load(ckpt_path)
new_ckpt = {}

# for 3DHPE
if 'model_pos' in ckpt.keys():
    new_ckpt['model_pos'] = ckpt['model_pos']
# for skeleton-based action recognition
elif 'model' in ckpt.keys():
    new_ckpt['model'] = ckpt['model']
else:
    assert NotImplementedError, 'For other motion tasks, our reload process are not programed!!'

torch.save(new_ckpt, ckpt_path + '.pth')
