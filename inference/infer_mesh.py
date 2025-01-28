import copy
import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))

from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import numpy as np

from inference.cfg.file_config import get_config

import torch

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# combine parameters from cmd and yaml
# cmd > yaml
def combine_cfg_and_args():
    motion_config_file = osp.join(r'cfg/mesh_config.yaml')
    cfg = get_config(motion_config_file)

    def get_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--vid_path', type=str, default='video/kunkun_cut.mp4', help='Wild video path.')
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


# get 2d keypoints from vitpose
def get_detector_2d(detector_name: str = 'vit_pose'):
    '''
    :param detector_name: str name of 2d dector's name (Currently, we only support 'vit_pos')
    :return:
    '''

    def get_vit_pose():
        from inference.infer_pose2d import inference_2d_keypoints_and_mesh_with_gen_npz as vit_pose
        return vit_pose

    detector_map = {
        'vit_pose': get_vit_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


# infer keypoints from trained model
def inference_mesh(cfg, keypoints_2d: np.array,
                   rot_vec: np.array, body_shape: np.array, root_transl: np.array, reg_keypoints_3d: np.array,
                   valid_bboxes: np.array,
                   vid_w: int = None, vid_h: int = None, vid_fps: int = 25):
    '''

    :param cfg: from config file
    :param keypoints_2d: [T,V,C], predicted by VitPose
    :param rot_vec: [T,V,C], predicted by HybrIK
    :param body_shape: [T,10], predicted by HybrIK
    :param root_transl: [T,C=3], predicted by HybrIK
    :param reg_keypoints_3d: [T,V,C=3], predicted by HybrIK
    :param valid_bboxes: [T,4], predicted bboxes
    :param vid_w: frame width
    :param vid_h: frame height
    :param vid_fps: fps for video
    :return:
        rot_vec_all: [N,T,V,C]
        body_shape_all: [N,T,10]
        world_root_transl: [N,T,C=3], in world space translation
        cam_root_transl: [N,T,C=3], in camera space translation
        projection_root_transl: [N,T,C=3], in projection space translation
    '''

    total_frame_cnt = keypoints_2d.shape[0]

    # use refine module
    if cfg.mesh.use_refine:
        from inference.tool.dataset_wild import WildDetNumpyDatasetMeshRefine
        from inference.tool.utils_data import flip_data, flip_thetas_batch
        import torch

        if cfg.mesh.checkpoint_path.endswith('.bin'):
            assert NotImplementedError, 'Currently, action inference by pytorch has not been implemented!!'
        elif cfg.mesh.checkpoint_path.endswith('.onnx'):
            import onnxruntime
            print(">>> Supported onnxruntime version: ", onnxruntime.__version__)
            print(">>> Supported Opset versions: ", onnxruntime.get_available_providers())
            from inference.tool.model import MeshRefine
            model_pos = MeshRefine(from_main_cfg=cfg, model_path=cfg.mesh.refine_checkpoint_path)
        else:
            assert NotImplementedError, 'Other formats of suffix checkpoint file for mesh refine are not supported now!!'

        print('>> Refining mesh')

        testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
        }
        if cfg.mesh.use_pixel_normalize:
            # Keep relative scale with pixel coornidates (we sue this branch currently)
            wild_dataset = WildDetNumpyDatasetMeshRefine(
                motion_2d=keypoints_2d, rot_vec=rot_vec, body_shape=body_shape,
                clip_len=cfg.mesh.clip_len, vid_size=(vid_w, vid_h),
                scale_range=None,
                focus=None,
                format='coco')
        else:
            # Scale to [-1,1]
            wild_dataset = WildDetNumpyDatasetMeshRefine(
                motion_2d=keypoints_2d, rot_vec=rot_vec, body_shape=body_shape,
                clip_len=cfg.MotionBERT.clip_len,
                vid_size=None, scale_range=[1, 1], focus=None, format='coco')

        test_loader = DataLoader(wild_dataset, **testloader_params)
        rot_vec_all = []
        body_shape_all = []
        vertices_all = []
        keypoint_3d_all = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                batch_motion2d_input = batch_input['motion2d']
                batch_rot_vec_input = batch_input['rot_vec']
                batch_body_shape_input = batch_input['body_shape']
                N, T = batch_motion2d_input.shape[:2]

                if torch.cuda.is_available():
                    batch_motion2d_input = batch_motion2d_input.cuda()

                if cfg.mesh.no_conf:
                    batch_motion2d_input = batch_motion2d_input[:, :, :, :2]

                if cfg.mesh.flip_augument:
                    # flip data for argument (we use this branch currently)
                    batch_motion2d_input_flip = flip_data(batch_motion2d_input)
                    batch_rot_vec_input_flip = flip_thetas_batch(batch_rot_vec_input)
                    rot_vec, shape, vertices, keypoint_3d = model_pos(batch_motion2d_input,
                                                                      batch_rot_vec_input,
                                                                      batch_body_shape_input)
                    rot_vec_flip, shape_flip, vertices_flip, keypoint_3d_flip = model_pos(batch_motion2d_input_flip,
                                                                                          batch_rot_vec_input_flip,
                                                                                          batch_body_shape_input)
                    keypoint_3d_flip = flip_data(keypoint_3d_flip)  # Flip back
                    keypoint_3d = (keypoint_3d_flip + keypoint_3d) / 2.0
                    rot_vec_flip = flip_thetas_batch(torch.from_numpy(rot_vec_flip)).cpu().numpy()

                    from scipy.spatial.transform import Rotation as R
                    rot_mat = R.from_rotvec(rot_vec.reshape([-1, 3])).as_matrix()
                    rot_mat_flip = R.from_rotvec(rot_vec_flip.reshape([-1, 3])).as_matrix()
                    rot_vec = R.from_matrix((rot_mat_flip + rot_mat) / 2.0).as_rotvec().reshape([N, T, -1, 3])

                else:
                    rot_vec, shape, vertices, keypoint_3d = model_pos(batch_motion2d_input, batch_rot_vec_input,
                                                                      batch_body_shape_input)

                rot_vec_all.append(rot_vec)
                body_shape_all.append(shape)
                vertices_all.append(vertices)
                keypoint_3d_all.append(keypoint_3d)

        # [N,T,...]
        rot_vec_all = np.concatenate(rot_vec_all, axis=1)
        body_shape_all = np.concatenate(body_shape_all, axis=1)
        vertices_all = np.concatenate(vertices_all, axis=1)
        keypoint_3d_all = np.concatenate(keypoint_3d_all, axis=1)
        # [T,C=3] -> [N=1,T,C=3]
        root_transl = np.expand_dims(root_transl, axis=0)

        # convert to valid frame
        rot_vec_all = rot_vec_all[:, :wild_dataset.total_frame()]
        body_shape_all = body_shape_all[:, :wild_dataset.total_frame()]
        vertices_all = vertices_all[:, :wild_dataset.total_frame()]
        keypoint_3d_all = keypoint_3d_all[:, :wild_dataset.total_frame()]

        if cfg.mesh.relat_root:
            vertices_all = vertices_all - vertices_all[:, 0, :, :]
            keypoint_3d_all = keypoint_3d_all - keypoint_3d_all[:, 0, :, :]
    # results directly from HybrIK
    else:
        rot_vec_all = np.expand_dims(rot_vec, axis=0)
        body_shape_all = np.expand_dims(body_shape, axis=0)
        keypoint_3d_all = np.expand_dims(reg_keypoints_3d, axis=0)

    # estimated camera if used and renew the root joint
    if cfg.mesh.motion_3d.use_estimated_camera:
        # for this method, world and camera space is None
        world_root_transl, cam_root_transl, projection_root_transl = None, None, None
        extrics_matrix_list, intrics_matrix_list = None, None

        # [T,C]
        renew_root_transl = camera_transformation_evaluation(cfg, keypoints_2d, keypoint_3d_all[0], root_transl[0],
                                                             vid_w, vid_h)
        # [T,C] -> [N=1,T,C]
        renew_root_transl = np.expand_dims(renew_root_transl, axis=0)
        # root_transl[:, :, :2] = renew_root_transl[:, :, :2]
        root_transl = renew_root_transl
    # for init hybrik camera space -> pixel change
    else:
        # camera option check
        intrinsics_camera_cfg = cfg.mesh.camera.intrinsics
        extrinsics_camera_cfg = cfg.mesh.camera.extrinsics
        assert intrinsics_camera_cfg.type.lower() in intrinsics_camera_cfg.support_type, "Other camera type doesn't support now!!"
        if intrinsics_camera_cfg.type.lower() == 'perspective':
            assert intrinsics_camera_cfg.principal.lower() in intrinsics_camera_cfg.support_principal_type, \
                "Ohter principal type for perspective camera doesn't support now!!"

        print('>> Converting from camera space into projection space...')
        # set basic pytorch3d path
        sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'tool/pytorch3d'))
        sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'tool/pytorch3d/pytorch3d'))
        import torch
        from inference.tool.pytorch3d.pytorch3d.renderer.cameras import PerspectiveCameras
        from inference.tool.pytorch3d.pytorch3d.transforms import Transform3d
        from tqdm import trange
        # from inference.tool.camera import camera_to_world
        from scipy.spatial.transform import Rotation as R

        def xyxy2xywh(bboxes):
            '''
            xyxy -> xywh
            :param bboxes: [T,4] or [N*T,4]
            :return: [T,4] or [N*T,4]
            '''
            x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            bboxes_converted = np.stack([cx, cy, w, h], axis=-1)

            # bboxes_converted = []
            # for bbox in bboxes:
            #     x1, y1, x2, y2 = bbox
            #
            #     cx = (x1 + x2) / 2
            #     cy = (y1 + y2) / 2
            #     w = x2 - x1
            #     h = y2 - y1
            #     bboxes_converted.append([cx, cy, w, h])
            # bboxes_converted = np.stack(bboxes_converted, axis=0)

            return bboxes_converted

        # init coordinate in camera space
        # [N=1,T,4]
        cam_root_transl = copy.deepcopy(root_transl)
        # # coordinate in world space
        # world_root_transl = camera_to_world(cam_root_transl,
        #                                     R=np.array(extrinsics_camera_cfg.R, dtype=np.float32),
        #                                     t=extrinsics_camera_cfg.T[0])

        # basic rotation matrix for world transfer -> camera space transfer
        # [1,3,3]
        cam_rotation_matrix = np.expand_dims(R.from_quat(extrinsics_camera_cfg.R).as_matrix(), axis=0)
        cam_transl_vec = np.expand_dims(np.array(extrinsics_camera_cfg.T, dtype=np.float32), axis=0)

        # coordinate in projection space
        # [T,4]
        bbox_xywh = xyxy2xywh(valid_bboxes)
        # [T,4] -> [N=1,T,4]
        bbox_xywh = np.expand_dims(bbox_xywh, axis=0)

        # for each frame, current is used to the projection space
        root_transl_world_sys_list = []
        root_transl_projection_sys_list = []
        extrics_matrix_list, intrics_matrix_list = [], []

        person_idx = 0
        for t_idx in trange(total_frame_cnt):
            focal = intrinsics_camera_cfg.focal_length
            cur_bbox = bbox_xywh[person_idx, t_idx]
            cur_root_transl = root_transl[person_idx, t_idx]
            cur_transl_camsys = copy.deepcopy(cur_root_transl)
            cur_transl_camsys = cur_transl_camsys * 256 / cur_bbox[2]

            focal = focal / 256 * cur_bbox[2]

            cameras = PerspectiveCameras(
                focal_length=((2 * focal / min(vid_h, vid_w), 2 * focal / min(vid_h, vid_w)),),
                # principal_point=((int(vid_w / 2), int(vid_h / 2)),),
                R=torch.from_numpy(cam_rotation_matrix), T=torch.from_numpy(cam_transl_vec)
            )

            camera_to_projectction_transform: Transform3d = cameras.get_projection_transform()
            camera_to_world_transform: Transform3d = cameras.get_world_to_view_transform().inverse()

            ###########################################################################
            # [1,4,4]
            world_to_camera_matrix = camera_to_world_transform.inverse().get_matrix().cpu().numpy()
            extrics_matrix_list.append(world_to_camera_matrix[0, :, :])

            # fake_world_to_camera_matrix = np.concatenate([cam_rotation_matrix[0], [[0], [0], [0]]], axis=1)
            # fake_world_to_camera_matrix = np.concatenate([fake_world_to_camera_matrix, [[0, 0, 0, 1]]], axis=0)
            # print(world_to_camera_matrix, fake_world_to_camera_matrix)
            # print('xxxxxxxxxxxxxxxxxxxxxx')
            ###########################################################################
            # [1,4,4] -> [1,3,3]
            camera_to_projection_matrix = camera_to_projectction_transform.get_matrix().cpu().numpy()
            intrics_matrix_list.append(camera_to_projection_matrix[0, :3, :3])
            ###########################################################################
            root_transl_world_res = camera_to_world_transform.transform_points(
                torch.from_numpy(cur_transl_camsys).reshape([-1, 3])
            )
            root_transl_world_res = root_transl_world_res.cpu().numpy()
            root_transl_world_sys_list.append(root_transl_world_res)

            ###########################################################################
            root_transl_projection_res = camera_to_projectction_transform.transform_points(
                torch.from_numpy(cur_transl_camsys).reshape([-1, 3])
            )
            root_transl_projection_res = root_transl_projection_res.cpu().numpy()
            root_transl_projection_sys_list.append(root_transl_projection_res)
            ###########################################################################

        ###################################################################################
        # get final result for world space and projection space
        # world space
        world_root_transl = np.concatenate(root_transl_world_sys_list, axis=0)
        # resotre into [N=1,T,C]
        world_root_transl = np.expand_dims(world_root_transl, axis=0)

        # print('xxxxxxxxxxxxxx', np.isclose(world_root_transl2, world_root_transl))
        # print('init')
        # print(world_root_transl.shape)
        # print('after')
        # print(world_root_transl2.shape)

        # projection space
        projection_root_transl = np.concatenate(root_transl_projection_sys_list, axis=0)
        # resotre into [N=1,T,C]
        projection_root_transl = np.expand_dims(projection_root_transl, axis=0)

    if cfg.mesh.render_result.save_render_frame:
        visualize_mesh(cfg=cfg, vid_w=vid_w, vid_h=vid_h,
                       rot_vec_all=rot_vec_all, body_shape_all=body_shape_all,
                       root_transl=projection_root_transl if projection_root_transl is not None else root_transl,
                       extrics_matrix_list=extrics_matrix_list, intrics_matrix_list=intrics_matrix_list)

    return rot_vec_all, body_shape_all, world_root_transl, cam_root_transl, projection_root_transl


# visualize smpl with Open3D tool
def visualize_mesh(cfg, vid_w, vid_h,
                   rot_vec_all, body_shape_all, root_transl,
                   extrics_matrix_list, intrics_matrix_list
                   ):
    assert osp.exists(cfg.mesh.render_result.smpl_parameter_path), "SMPL parameter path doesn't exist!!"
    print('>> Start render by Open3D')
    from inference.tool.nosmpl.tools.smpl_visualizer import SMPLVisualizer

    save_dir = osp.join('output_mesh', osp.basename(cfg.video_path).split('.')[0] + '_mesh_smpl')
    visualizer = SMPLVisualizer(fps=vid_fps, frame_width=vid_w, frame_height=vid_h,
                                smpl_parameter_path=cfg.mesh.render_result.smpl_parameter_path,
                                export_mp4=cfg.mesh.render_result.export_gif,
                                export_gif=cfg.mesh.render_result.export_gif,
                                tmp_img_dir=save_dir,
                                render_video_path=video_file_path)

    visualizer.forward(rot_vec=rot_vec_all, pred_shape=body_shape_all,
                       root_transl=root_transl,
                       # extrinsics_matrix_list=extrics_matrix_list,
                       # intrinsics_matrix_list=intrics_matrix_list
                       )

    print('>> Render successfully')


# camera estiamtion function solver (have error currently)
def camera_transformation_evaluation(cfg, keypoints_2d: np.array,
                                     reg_keypoints3d: np.array, root_transl: np.array,
                                     vid_w: int = None, vid_h: int = None):
    '''

    :param cfg:
    :param keypoints_2d: [T,V,C]
    :param reg_keypoints3d: [T,V,C]
    :param root_transl: [T,C=3]
    :param vid_w:
    :param vid_h:
    :return: renewd_root_transl [T,C]
    '''
    import torch
    from inference.tool.dataset_wild import WildDetNumpyDataset
    from inference.tool.utils_data import flip_data

    def solve_scale(x, y):
        def err(p, x, y):
            return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()

        from scipy.optimize import least_squares
        print('>> Estimating camera transformation.')
        best_res = 100000
        best_scale = None
        for init_scale in tqdm(range(0, 2000, 5)):
            p0 = [init_scale, 0.0, 0.0, 0.0]
            est = least_squares(err, p0, args=(x.reshape(-1, 3), y.reshape(-1, 3)))
            if est['fun'] < best_res:
                best_res = est['fun']
                best_scale = est['x'][0]
        print('>> Pose matching error = %.2f mm.' % best_res)
        return best_scale

    print('>> Estimating camera transformation')
    # 1. estimate 3DHPE
    print('>> Estimating 3D human pose')
    from inference.tool.model import PoseEstimator
    model_pos = PoseEstimator(from_main_cfg=cfg, model_path=cfg.mesh.motion_3d.checkpoint_path)

    testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
    }

    if cfg.mesh.motion_3d.use_pixel_normalize:
        # Keep relative scale with pixel coornidates (we sue this branch currently)
        wild_dataset = WildDetNumpyDataset(numpy_data=keypoints_2d,
                                           clip_len=cfg.mesh.motion_3d.clip_len, vid_size=(vid_w, vid_h),
                                           scale_range=None,
                                           focus=None,
                                           format='coco')
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetNumpyDataset(numpy_data=keypoints_2d, clip_len=cfg.mesh.motion_3d.clip_len,
                                           vid_size=None, scale_range=[1, 1], focus=None, format='coco')

    test_loader = DataLoader(wild_dataset, **testloader_params)
    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            if cfg.mesh.motion_3d.no_conf:
                batch_input = batch_input[:, :, :, :2]

            if cfg.mesh.motion_3d.flip_augument:
                # flip data for argument (we use this branch currently)
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)

            if cfg.mesh.motion_3d.rootrel:
                # In this mode, the root joint is all around 0
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                # In this mode, the start time's z-axith is 0
                # we use this branch currently
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass

            # not used currently
            if cfg.mesh.motion_3d.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]

            if isinstance(predicted_3d_pos, torch.Tensor):
                predicted_3d_pos = predicted_3d_pos.cpu().numpy()
            results_all.append(predicted_3d_pos)

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    results_all = results_all[:wild_dataset.total_frame()]

    if cfg.mesh.motion_3d.use_pixel_restore:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0
    prediction_world_3d = results_all

    # 2. solve camera for root joint ([T,C])
    x = prediction_world_3d - prediction_world_3d[:, :1]
    y = reg_keypoints3d - reg_keypoints3d[:, :1]
    scale = solve_scale(x, y)
    predicted_camera_root = prediction_world_3d[:, 0] * scale

    # root_transl = root_transl - reg_keypoints3d[:, 0] + predicted_camera_root
    root_transl = predicted_camera_root
    return root_transl


# save fbx or bvh in Blender tools
def save_fbx_and_bvh(cfg, vid_fps, rot_vec, root_transl):
    # there lies 2 way to use byp: (1) directly install bpy (2) use Blender environment
    # check if there lies bpy package, it will use it, else use installed blender root path for fbx and bvh output
    use_bpy = True
    try:
        import bpy
    except ModuleNotFoundError:
        use_bpy = False
        # add temp blender home dir into system path if not exists bpy package
        assert osp.exists(
            cfg.mocap.blender_home), f'Your binding blender environment root path {cfg.mocap.blender_home} should be exists!!'
        current_env = os.environ.get('Path')
        new_path = f'{current_env};{cfg.mocap.blender_home}'
        os.environ['Path'] = new_path

    from collections import defaultdict
    from typing import DefaultDict, List
    from numpy import typing as npt

    print('>> Saving intermediate SMPL parameter parameter')
    # [T,72]
    rotations_all: npt.NDArray[np.float32] = rot_vec.reshape([-1, 72])
    # [T,3]
    root_trans: npt.NDArray[np.float32] = root_transl.reshape([-1, 3])

    total_frame = root_trans.shape[0]

    frame_names: List[int] = [item + 1 for item in range(total_frame)]
    frame_results: DefaultDict[int, DefaultDict[str, npt.NDArray[np.float32]]] = defaultdict()
    for t_index, frame_name in enumerate(frame_names):
        frame_results[frame_name] = defaultdict()
        frame_results[frame_name]['smpl_thetas'] = np.expand_dims(rotations_all[t_index], axis=0)
        frame_results[frame_name]['cam_trans'] = np.expand_dims(root_trans[t_index], axis=0)

    # in single person mode, this value is empty
    sequence_results = []

    save_path = osp.join('output_mesh', osp.basename(cfg.video_path).split('.')[0] + '_mesh_smpl.npz')

    np.savez_compressed(save_path,
                        results=frame_results,
                        sequence_results=sequence_results
                        )

    print(f'>> Saved *.npz file at {save_path} successfully')

    cfg.mocap.bpy.input_path = save_path
    cfg.mocap.bpy.fps_target = cfg.mocap.bpy.fps_source = int(vid_fps)
    cfg.mocap.bpy.root_path = cur_file_absolute_path = osp.abspath(osp.dirname(__file__))

    if cfg.mocap.bvh_export:
        print('>> Start converting for bvh')
        output_bvh_path = osp.join('output_mesh', osp.basename(cfg.video_path).split('.')[0] + '_mesh_mocap.bvh')
        cfg.mocap.bpy.output_path = output_bvh_path

        if use_bpy:
            from inference.tool.convert2fbx import external_main as export_extension_func
            export_extension_func(cfg.mocap.bpy)
        else:
            try:
                import subprocess
                input_blender_command = f"blender {osp.join(cur_file_absolute_path, cfg.mocap.empty_blender_file)} " \
                                        f"--background " \
                                        f"--python {osp.join(cur_file_absolute_path, cfg.mocap.bpy.target_script_path)} " \
                                        f"--use_relative_path True " \
                                        f"--root_path {cur_file_absolute_path} " \
                                        f"--input_path {cfg.mocap.bpy.input_path} " \
                                        f"--output_path {cfg.mocap.bpy.output_path} " \
                                        f"--gender {cfg.mocap.bpy.gender} " \
                                        f"--fps_source {cfg.mocap.bpy.fps_source} " \
                                        f"--fps_target {cfg.mocap.bpy.fps_target} " \
                                        f"--male_model_path {cfg.mocap.bpy.male_model_path} " \
                                        f"--female_model_path {cfg.mocap.bpy.female_model_path} " \
                                        f"--character_model_path {cfg.mocap.bpy.character_model_path}"
                print(input_blender_command)
                proc = subprocess.Popen(input_blender_command)
                proc.wait()
            except BaseException:
                print('>> Error then exit!!')
                exit(-1)

        from inference.tool.bvh.modifier.bvhio import bvhio
        bvh_motion = bvhio.readAsHierarchy(output_bvh_path)
        frame_time = bvhio.readAsBvh(output_bvh_path).FrameTime
        root_joint = bvh_motion.filter(cfg.mocap.root_joint_name)[0]
        new_root_joint = root_joint.clearParent(keep=['position', 'rotation', 'scale', 'rest', 'anim'])
        output_bvh_single_person_path = osp.join('output_mesh',
                                                 osp.basename(cfg.video_path).split('.')[0] + '_mesh_mocap_single.bvh')
        bvhio.writeHierarchy(output_bvh_single_person_path, new_root_joint, frame_time)

        print('>> Convert bvh successfully')

    if cfg.mocap.fbx_export:
        print('>> Start converting for fbx')
        output_fbx_path = osp.join('output_mesh', osp.basename(cfg.video_path).split('.')[0] + '_mesh_mocap.fbx')
        cfg.mocap.bpy.output_path = output_fbx_path

        if use_bpy:
            from inference.tool.convert2fbx import external_main as export_extension_func
            export_extension_func(cfg.mocap.bpy)
        else:
            try:
                import subprocess
                input_blender_command = f"blender {osp.join(cur_file_absolute_path, cfg.mocap.empty_blender_file)} " \
                                        f"--background " \
                                        f"--python {osp.join(cur_file_absolute_path, cfg.mocap.bpy.target_script_path)} " \
                                        f"--use_relative_path True " \
                                        f"--root_path {cur_file_absolute_path} " \
                                        f"--input_path {cfg.mocap.bpy.input_path} " \
                                        f"--output_path {cfg.mocap.bpy.output_path} " \
                                        f"--gender {cfg.mocap.bpy.gender} " \
                                        f"--fps_source {cfg.mocap.bpy.fps_source} " \
                                        f"--fps_target {cfg.mocap.bpy.fps_target} " \
                                        f"--male_model_path {cfg.mocap.bpy.male_model_path} " \
                                        f"--female_model_path {cfg.mocap.bpy.female_model_path} " \
                                        f"--character_model_path {cfg.mocap.bpy.character_model_path}"
                print(input_blender_command)
                proc = subprocess.Popen(input_blender_command)
                proc.wait()
            except BaseException:
                print('>> Error then exit!!')
                exit(-1)

        print('>> Convert fbx successfully')

    print('>> All files export successfully!!')


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
        keypoints_2d, keypoints_2d_score, rot_vec, body_shape, root_transl, pred_keypoints17, valid_bboxes = \
            detector_2d(video_path, cfg)
    else:
        # assert NotImplementedError, 'Currently, directly read 2d kyepoints from npz file is not supported!!'
        keypoints_tuple_with_mesh = np.load(cfg.detector_2d.input_npz, allow_pickle=True)
        keypoints_2d, keypoints_2d_score = keypoints_tuple_with_mesh['kpts'], keypoints_tuple_with_mesh['kpts_score']
        rot_vec, body_shape, root_transl, pred_keypoints17 = keypoints_tuple_with_mesh['rot_vec'], \
                                                             keypoints_tuple_with_mesh['body_shape'], \
                                                             keypoints_tuple_with_mesh['root_transl'], \
                                                             keypoints_tuple_with_mesh['pred_keypoints17']
        valid_bboxes = keypoints_tuple_with_mesh['bbox']

    print('>> Successfully predicted 2D keypoints and mesh information!!')

    # 2.  mesh inference and refine
    # if needed, visualize them
    keypoints_2d_combine = np.concatenate([keypoints_2d, keypoints_2d_score], axis=2)
    rot_vec_all, body_shape_all, world_root_transl, cam_root_transl, proj_root_transl = inference_mesh(cfg=cfg,
                                                                                                       keypoints_2d=keypoints_2d_combine,
                                                                                                       vid_w=vid_w,
                                                                                                       vid_h=vid_h,
                                                                                                       vid_fps=vid_fps,
                                                                                                       rot_vec=rot_vec,
                                                                                                       body_shape=body_shape,
                                                                                                       root_transl=root_transl,
                                                                                                       reg_keypoints_3d=pred_keypoints17,
                                                                                                       valid_bboxes=valid_bboxes)
    print('>> Successfully refined mesh!!')

    # 3. convert fbx or bvh(if needed)
    if cfg.mocap.fbx_export or cfg.mocap.bvh_export:
        save_fbx_and_bvh(cfg, vid_fps, rot_vec_all, proj_root_transl)


if __name__ == '__main__':
    # load config from certain *.yaml config file and cmd
    cfg = combine_cfg_and_args()
    video_file_path = cfg.video_path
    _, vid_fps, vid_size = get_vid_info(video_file_path)
    main(cfg, vid_size, vid_fps)
