import os
import os.path as osp
import sys

sys.path.append(osp.join('..'))

import easydict
from inference.vitpose.inference_pose import PoseEstimator, PoseEstimatorWithMesh

vitpose_root_path = osp.join(r'vitpose/easy_ViTPose')


def inference_2d_keypoints_with_gen_npz(video_path, main_cfg):
    '''
    :param video_path: video source path
    :param main_cfg: from mian config file parameters
    :return:
        {
            kpts_2d: [T,V=17,C=2],
            kpts_score: [T,V=17,C=1],
        }
    '''
    pose_estimator = PoseEstimator(from_main_cfg=main_cfg)
    ret_kpt_and_score = easydict.EasyDict({'kpt': None, 'score': None})
    pose_estimator.inference(ret_kpt_and_score=ret_kpt_and_score)
    kpts_2d, kpts_score = ret_kpt_and_score.kpt, ret_kpt_and_score.score
    if main_cfg.detector_2d.estimation_result.save_npz:
        import numpy as np
        save_npz_dir = osp.join('output_2dhpe_npz')
        os.makedirs(save_npz_dir, exist_ok=True)
        output_path_npz = osp.join(save_npz_dir, osp.basename(video_path).split('.')[0] + '_2dhpe.npz')
        print('>>> Keypoint2d and keypoint_2d npz save in ', output_path_npz)
        np.savez_compressed(output_path_npz, kpts=kpts_2d, kpts_score=kpts_score)
    return kpts_2d, kpts_score


def inference_2d_keypoints_and_mesh_with_gen_npz(video_path, main_cfg):
    '''
    :param video_path: video source path
    :param main_cfg: from mian config file parameters
    :return:
        {
            kpts_2d: [T,V=17,C=2],
            kpts_score: [T,V=17,C=1],
            rot_vec: [T,V=24,C=3],
            body_shape: [T,10],
            bbox: [T,4],
            root_transl: [T,C=3], for root joint translation in camera space
            pred_keypoints17: [T,V,C] regressed 17 joints
        }
    '''
    pose_estimator = PoseEstimatorWithMesh(from_main_cfg=main_cfg)
    ret_kpt_and_score_with_mesh = easydict.EasyDict(
        {'kpt': None, 'score': None,
         'rot_vec': None, 'body_shape': None, 'root_transl': None, 'pred_keypoints17': None, 'bbox': None}
    )
    # pose_estimator.inference_strictly(ret_kpt_and_score_with_mesh=ret_kpt_and_score_with_mesh)
    pose_estimator.inference_loose(ret_kpt_and_score_with_mesh=ret_kpt_and_score_with_mesh)
    kpts_2d, kpts_score = ret_kpt_and_score_with_mesh.kpt, ret_kpt_and_score_with_mesh.score
    rot_vec, body_shape, root_transl, pred_keypoints17, valid_bboxes = \
        ret_kpt_and_score_with_mesh.rot_vec, ret_kpt_and_score_with_mesh.body_shape, \
        ret_kpt_and_score_with_mesh.root_transl, ret_kpt_and_score_with_mesh.pred_keypoints17, \
        ret_kpt_and_score_with_mesh.bbox

    if main_cfg.detector_2d.estimation_result.save_npz:
        import numpy as np
        save_npz_dir = osp.join('output_2dhpe_npz')
        os.makedirs(save_npz_dir, exist_ok=True)
        output_path_npz = osp.join(save_npz_dir, osp.basename(video_path).split('.')[0] + '_2dhpe_with_mesh.npz')
        print('>>> Keypoint2d and keypoint_2d npz save in ', output_path_npz)
        np.savez_compressed(output_path_npz, kpts=kpts_2d, kpts_score=kpts_score,
                            rot_vec=rot_vec, body_shape=body_shape, root_transl=root_transl,
                            pred_keypoints17=pred_keypoints17, bbox=valid_bboxes)

    return kpts_2d, kpts_score, rot_vec, body_shape, root_transl, pred_keypoints17, valid_bboxes
