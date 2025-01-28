import logging
import os.path as osp
import sys

sys.path.append(osp.abspath(osp.dirname(__file__)))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl


def get_action_names(file_path="data/action/ntu_actions.txt"):
    f = open(file_path, "r")
    s = f.read()
    actions = s.split('\n')
    action_names = []
    for a in actions:
        action_names.append(a.split('.')[1][1:])
    return action_names


# 将样本中的所有坐标都归一化后并从[0,1]->[-1,1]
def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam


# 将coco格式的16个关键点转换为H36M格式的关键点格式
def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
    y[:, :, 1, :] = x[:, :, 12, :]
    y[:, :, 2, :] = x[:, :, 14, :]
    y[:, :, 3, :] = x[:, :, 16, :]
    y[:, :, 4, :] = x[:, :, 11, :]
    y[:, :, 5, :] = x[:, :, 13, :]
    y[:, :, 6, :] = x[:, :, 15, :]
    y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
    y[:, :, 9, :] = x[:, :, 0, :]
    y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5
    y[:, :, 11, :] = x[:, :, 5, :]
    y[:, :, 12, :] = x[:, :, 7, :]
    y[:, :, 13, :] = x[:, :, 9, :]
    y[:, :, 14, :] = x[:, :, 6, :]
    y[:, :, 15, :] = x[:, :, 8, :]
    y[:, :, 16, :] = x[:, :, 10, :]
    return y


# 这个是针对动作识别的数据增强，要求针对每一个时刻所有点都做随机的扰动
# 在实际应用中，除了data_numpy,其他的最终选用的是默认参数
def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))  # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    # 从move_time_candidate中随机选择一个数，这里因为是默认值，所以只能选 1
    # 最终的move_time表示采样点的插值数量
    move_time = random.choice(move_time_candidate)
    # 选择从 [0,T) 的点数,采样间隔为 T * 1.0 / move_time
    # 选择随机采样点的下标,最终选择的数值为 [0,T*deta*1,T*deta*2,...,T),其中deta=T*1.0/move_time
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    # 再加入最终的尾部截止时刻T
    node = np.append(node, T)
    # num_node表示所有要采样点的数量
    num_node = len(node)
    # 随机角度旋转,初始化为 num_node,默认为T
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    # 随机缩放,初始化为 num_node,默认为T
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    # 随机平移,初始化为 num_node,默认为T
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    # 对所有时刻的旋转、缩放和平移参数做列表的存储
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    # 根据线性插值生成所有的时刻变换的对应点
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    # thea的shape为 [2,2,T]
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # 将变换作用到xy上，生成结果存储在new_xy上
    for i_frame in range(T):
        # data_numpy的shape为 [V,T,C,M],xy的shape为 [V'=2,C,M](看成由多人拼接成的列向量)
        xy = data_numpy[0:2, i_frame, :, :]
        # 对于np.dot,如果np.dot(a,b)中有任意一个为矩阵时，这个np.dot就为矩阵乘法
        # 而当a和b都为向量时，这个np.dot就为向量的点积
        # 在做旋转矩阵乘法之前，这里的xy要reshape为 [V'=2,C*M]
        # 最后得到的new_xy shape 为 [2,C*M]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))  # C,T,V,M -> M,T,V,C
    return data_numpy


# 个人理解，这段函数主要负责单人或多人的轨迹追踪
# 这个函数用于解决在人物追踪过程中可能出现的两个人关键点（坐标点）重叠的情况。
# 当两个人物的关键点非常接近或者在某些时间步骤上出现重叠时，这个函数会尝试根据
# 运动情况来重新分配这些关键点，以判断它们是否属于同一个人物
def human_tracking(x):
    M, T = x.shape[:2]
    if M == 1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0, 1:] - x[0, :-1], axis=-1), axis=-1)  # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0, 1:] - x[1, :-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        # 通过比较两个人物轨迹的运动情况，确定在每个时间步选择哪个人物的轨迹。sel是一个一维数组，元素值为0或1，用来表示选择哪个人物
        # 关于这个 np.cumsum 函数，用来做内部数组的累加求和
        # 例： np.cumsum([1,2,3,0,5]) -> [1,3(=1+2),6(=3+3),6(=6+0),11(=6+5)]
        # 理论上来说,diff0<diff1时，我们可以认为这个轨迹更偏向 第一个人，因此这时的点属于第0个人
        # 当 diff0>diff1时，我们可以认为这个轨迹更偏向 第二个人，因此这时的点属于第1个人
        sel = np.cumsum(diff0 > diff1) % 2
        # 将sel拓展为三维
        sel = sel[:, None, None]
        # 对所有点做重新的拟合
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0, 1:] = x[1, 1:] * sel + x[0, 1:] * (1 - sel)
        x_new[1, 1:] = x[0, 1:] * sel + x[1, 1:] * (1 - sel)
        return x_new


class ActionDataset(Dataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1, 1],
                 check_split=True):  # data_split: train/test etc.
        '''
        在训练的时候 random_move 为 True, Test时 为 False
        scale_range的范围 为
            scale_range_train: [1, 3]
            scale_range_test: [2, 2]
        '''
        np.random.seed(0)
        dataset = read_pkl(data_path)
        if check_split:
            assert data_split in dataset['split'].keys()
            self.split = dataset['split'][data_split]
        annotations = dataset['annotations']
        self.random_move = random_move
        # 在test时这个值为False，在Train时这个值为True
        self.is_train = "train" in data_split or (check_split == False)
        # 小样本学习我们这边不需要，不要管
        if "oneshot" in data_split:
            self.is_train = False
        self.scale_range = scale_range
        motions = []
        labels = []
        for sample in annotations:
            if check_split and (not sample['frame_dir'] in self.split):
                continue
            # 做数据重采样，获得所有待重采样的下标
            resample_id = resample(ori_len=sample['total_frames'], target_len=n_frames, randomness=self.is_train)
            # 将样本中的所有坐标都归一化后并从[0,1]->[-1,1]
            motion_cam = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])
            # 个人理解，这段函数的作用是由于用HRNet做关键点检测时得到的多人关键点存在问题(在NTU-RGB这个数据集中多人至多为2个人)，因此
            # 这里需要用这个函数来做一下重新粗略的匹配(根据轨迹做粗略的匹配)
            motion_cam = human_tracking(motion_cam)
            # 将coco格式转换为h36m关键点格式 输入为 [M,T,V,C]
            motion_cam = coco2h36m(motion_cam)
            # sample['keypoint_score'] 为 [M,T,V],因此需要拓展出最后一个通道维度变为 [M,T,V,C=1]
            motion_conf = sample['keypoint_score'][..., None]
            # 最后拼接为 [x,y,conf_z] 这样的 [M,T,V,C=3]
            motion = np.concatenate((motion_cam[:, resample_id], motion_conf[:, resample_id]), axis=-1)
            # 这里由于NTU-RGB+D是最多两个人的，因此空闲的人用zero_padding策略做补齐
            if motion.shape[0] == 1:  # Single person, make a fake zero person
                fake = np.zeros(motion.shape)
                motion = np.concatenate((motion, fake), axis=0)
            motions.append(motion.astype(np.float32))
            labels.append(sample['label'])
        # 所有的标签和动作信息都放在list中
        self.motions = np.array(motions)
        self.labels = np.array(labels)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, index):
        raise NotImplementedError


# 在训练和测试两个数据集的加载器上有如下的设置：
# 训练阶段的数据加载器
# ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split + '_train', n_frames=args.clip_len,
#                                random_move=args.random_move, scale_range=args.scale_range_train)
# 评估阶段的数据加载器
# ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split + '_val', n_frames=args.clip_len,
#                              random_move=False, scale_range=args.scale_range_test)
class NTURGBD(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1, 1], action_label_map=''):
        super(NTURGBD, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        self.action_label_map = [x.strip() for x in
                                 open(osp.join(action_label_map)).readlines()] if action_label_map != '' else None

    # 返回值为两个参数，分别为经过处理后的motion [-1,1] 数据和标签label
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx]  # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion

        # label map is specially for vis_attn
        if self.action_label_map is not None:
            return result.astype(np.float32), label, self.action_label_map[label]
        else:
            return result.astype(np.float32), label


class NTURGBD1Shot(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1, 1], check_split=False):
        super(NTURGBD1Shot, self).__init__(data_path, data_split, n_frames, random_move, scale_range, check_split)
        oneshot_classes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        new_classes = set(range(120)) - set(oneshot_classes)
        old2new = {}
        for i, cid in enumerate(new_classes):
            old2new[cid] = i
        filtered = [not (x in oneshot_classes) for x in self.labels]
        self.motions = self.motions[filtered]
        filtered_labels = self.labels[filtered]
        self.labels = [old2new[x] for x in filtered_labels]

    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx]  # (M,T,J,C)
        if self.random_move:
            # 针对每一个时刻所有点都做随机的扰动
            # 经过扰动后，此时[x,y,conf_z]中只有conf_z在多人经过zero_padding的情况下为0，而x和y可能不会为0了
            motion = random_move(motion)
        if self.scale_range:
            # 将所有坐标做缩放，然后转换到[-1,1]中
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label


if __name__ == '__main__':
    # # 测试一下random.choice 这个函数表示从list中随机算则一个数
    # l = [1]
    # move_time = random.choice(l)
    # T = 100
    # node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    # node = np.append(node, T)
    # print(node)

    # 测试random_move这个函数
    M = 1
    C = 3
    T = 20
    V = 17
    tensor = np.random.randn(M, T, V, C)
    random_move(data_numpy=tensor)
