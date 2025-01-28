import os
import sys
import pickle
import numpy as np
import random

sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_h36m_tds import DataReaderH36M
from tqdm import tqdm


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-src_pkl_dir_root', type=str,
                        default=r'G:\3dhpe_dataset\motionbert_human3.6m\h36m_sh_conf_cam_source_final',
                        help='The root paht for source pickle directory.')
    parser.add_argument('-out_dir', type=str, default=r'data/motion3d/H36M-SH/Mixste_Style_v2',
                        help="The output save root dir for train and test dataset. Here, please attention the root dir here is your project's root path.")
    parser.add_argument('-tds', type=int, default=2,
                        help='Temporal downsampling for both train and test dataset.Here, we use 2 as default setting.')
    parser.add_argument('-f', type=int, default=243, help='The input sequence length.')
    parser.add_argument('-s', type=int, default=81,
                        help='Data sample stride strategy in train stage. Except MixSTE, other method use overlaped data sampling strategy.')
    return parser.parse_args()


args = get_args()


def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)


# This is for MixSTE (Nonoverlapping data sample stragegy)
# assert args.f == args.s, 'In MixSTE style, the sample step should be same.'
datareader = DataReaderH36M(n_frames=args.f, sample_stride=args.tds,
                            data_stride_train=args.s, data_stride_test=args.f,
                            dt_file='h36m_sh_conf_cam_source_final.pkl', dt_root=args.src_pkl_dir_root)
# downsampled data and label acquire
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
print(train_data.shape, test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

root_path = args.out_dir
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels)
save_clips("test", root_path, test_data, test_labels)
