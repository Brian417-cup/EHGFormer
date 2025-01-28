import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))

from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import numpy as np

from inference.cfg.file_config import get_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# combine parameters from cmd and yaml
# cmd > yaml
def combine_cfg_and_args():
    motion_config_file = osp.join(r'cfg/action_config.yaml')
    cfg = get_config(motion_config_file)

    def get_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--vid_path', type=str, default='video/clapping_hand.avi', help='Wild video path.')
        args = parser.parse_args()
        return args

    args = get_args()
    cfg.video_path = args.vid_path
    return cfg


# get video info
def get_vid_info(vid_path: str):
    vid = imageio.get_reader(vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    return vid, fps_in, vid_size


# ge
# t 2d keypoints from vitpose
def get_detector_2d(detector_name: str = 'vit_pose'):
    '''
    :param detector_name: str name of 2d dector's name (Currently, we only support 'vit_pos')
    :return:
    '''

    def get_vit_pose():
        from inference.infer_pose2d import inference_2d_keypoints_with_gen_npz as vit_pose
        return vit_pose

    detector_map = {
        'vit_pose': get_vit_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


# infer keypoints from trained model
def inference_action(cfg, keypoints_2d: np.array, vid_w: int = None, vid_h: int = None, vid_fps: int = 25):
    from inference.tool.dataset_wild import WildDetDatasetNTURGBDSingle
    import torch

    if cfg.action.checkpoint_path.endswith('.bin'):
        assert NotImplementedError, 'Currently, action inference by pytorch has not been implemented!!'
    elif cfg.action.checkpoint_path.endswith('.onnx'):
        import onnxruntime
        print(">>> Supported onnxruntime version: ", onnxruntime.__version__)
        print(">>> Supported Opset versions: ", onnxruntime.get_available_providers())
        from inference.tool.model import ActionRecognizer
        model = ActionRecognizer(from_main_cfg=cfg, model_path=cfg.action.checkpoint_path)
    else:
        assert NotImplementedError, 'Other formats of suffix checkpoint file for 3D pose are not supported now!!'

    testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
    }

    if cfg.action.use_pixel_normalize:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDatasetNTURGBDSingle(numpy_data=keypoints_2d,
                                                   clip_len=cfg.action.clip_len, vid_size=(vid_w, vid_h),
                                                   scale_range=None,
                                                   focus=None,
                                                   format='coco')
    else:
        # Scale to [-1,1] (We sue this branch currently)
        wild_dataset = WildDetDatasetNTURGBDSingle(numpy_data=keypoints_2d, clip_len=cfg.action.clip_len,
                                                   vid_size=(vid_w, vid_h), scale_range=[1, 1], focus=None,
                                                   format='coco')

    test_loader = DataLoader(wild_dataset, **testloader_params)
    label_map = [x.strip() for x in open(osp.join(cfg.action.label_map)).readlines()]

    from typing import List

    top5_results_all: List[str] = []
    top1_results_all: List[str] = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]

            assert N == 1, 'Currently, we only support for one video pertime!!'

            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            if cfg.action.no_conf:
                batch_input = batch_input[:, :, :, :2]

            predicted_output = model(batch_input)
            if isinstance(predicted_output, torch.Tensor):
                predicted_output = predicted_output.cpu().numpy()
            # caculate action recognition results
            # top5 caculate
            top5, top1 = 5, 1
            sorted_args = np.argsort(-predicted_output, axis=1)
            top5_labels_index = sorted_args[:, :top5]
            top1_labels_index = sorted_args[:, :top1]

            top5_results_all += [label_map[label_index] for label_index in top5_labels_index[0]]
            top1_results_all += [label_map[label_index] for label_index in top1_labels_index[0]]

    return top5_results_all, top1_results_all


# get 2d results and 3d results
def main(cfg, vid_size, vid_fps):
    vid_w, vid_h = vid_size

    # 1.acquire 2d detectors
    detector_2d = get_detector_2d(cfg.detector_2d.selected_name)

    # acquire 2d keypoints from 2d keypoint detectors directly or from npz file
    if cfg.detector_2d.input_npz == '':
        video_path = cfg.video_path
        video_name = osp.basename(video_path).split('.')[0]
        # [T,V,2] and [T,V,1]  C=2+1 , here x and y is not normalized!!
        keypoints_2d, keypoints_2d_score = detector_2d(video_path, cfg)
    else:
        # assert NotImplementedError, 'Currently, directly read 2d kyepoints from npz file is not supported!!'
        keypoints_tuple = np.load(cfg.detector_2d.input_npz, allow_pickle=True)
        keypoints_2d, keypoints_2d_score = keypoints_tuple['kpts'], keypoints_tuple['kpts_score']

    print('>> Successfully predicted 2D keypoints!!')

    # 2.  action recognition
    keypoints_2d_combine = np.concatenate([keypoints_2d, keypoints_2d_score], axis=2)
    action_results_top5, action_result_top1 = inference_action(cfg=cfg, keypoints_2d=keypoints_2d_combine,
                                                               vid_w=vid_w, vid_h=vid_h, vid_fps=vid_fps)
    print('>> Successfully predicted action!!')
    print(
        f'>> All top5 possible action results are {action_results_top5}, and top1 action results is {action_result_top1}')


if __name__ == '__main__':
    # load config from certain *.yaml config file and cmd
    cfg = combine_cfg_and_args()
    video_file_path = cfg.video_path
    _, vid_fps, vid_size = get_vid_info(video_file_path)
    main(cfg, vid_size, vid_fps)
