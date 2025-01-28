import os
import os.path as osp
import sys

import torch

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))

from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm, trange
import numpy as np
import shutil

from inference.cfg.file_config import get_config

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# combine parameters from cmd and yaml
# cmd > yaml
def combine_cfg_and_args():
    motion_config_file = osp.join(r'cfg/mocap_config.yaml')
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
        from inference.infer_pose2d import inference_2d_keypoints_with_gen_npz as vit_pose
        return vit_pose

    detector_map = {
        'vit_pose': get_vit_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


# infer keypoints from trained model
def inference_pose3d(cfg, keypoints_2d: np.array, vid_w: int = None, vid_h: int = None, vid_fps: int = 25):
    from inference.tool.dataset_wild import WildDetNumpyDataset
    from inference.tool.utils_data import flip_data
    import torch

    if cfg.detector_3d.checkpoint_path.endswith('.bin'):
        from inference.tool.learning import load_backbone
        import torch
        torch.multiprocessing.freeze_support()

        backbone_cfg = get_config(cfg.detector_3d.model_config_path)
        model_backbone = load_backbone(backbone_cfg)
        if torch.cuda.is_available():
            model_backbone = model_backbone.cuda()
        print('>>> Loading checkpoint')
        ckpt = torch.load(cfg.detector_3d.checkpoint_path, map_location=lambda storage, loc: storage)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['model_pos'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # multi gpu -> single gpu
        model_backbone.load_state_dict(new_state_dict, strict=True)
        model_pos = model_backbone
        model_pos.eval()
    elif cfg.detector_3d.checkpoint_path.endswith('.onnx'):
        import onnxruntime
        print(">>> Supported onnxruntime version: ", onnxruntime.__version__)
        print(">>> Supported Opset versions: ", onnxruntime.get_available_providers())
        from inference.tool.model import PoseEstimator
        model_pos = PoseEstimator(from_main_cfg=cfg, model_path=cfg.detector_3d.checkpoint_path)
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
    if cfg.detector_3d.use_pixel_normalize:
        # Keep relative scale with pixel coornidates (we sue this branch currently)
        wild_dataset = WildDetNumpyDataset(numpy_data=keypoints_2d,
                                           clip_len=cfg.detector_3d.clip_len, vid_size=(vid_w, vid_h),
                                           scale_range=None,
                                           focus=None,
                                           format='coco')
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetNumpyDataset(numpy_data=keypoints_2d, clip_len=cfg.MotionBERT.clip_len,
                                           vid_size=None, scale_range=[1, 1], focus=None, format='coco')

    test_loader = DataLoader(wild_dataset, **testloader_params)
    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            if cfg.detector_3d.no_conf:
                batch_input = batch_input[:, :, :, :2]

            if cfg.detector_3d.flip_augument:
                # flip data for argument (we use this branch currently)
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)

            if cfg.detector_3d.rootrel:
                # In this mode, the root joint is all around 0
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                # In this mode, the start time's z-axith is 0
                # we use this branch currently
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass

            # not used currently
            if cfg.detector_3d.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]

            if isinstance(predicted_3d_pos, torch.Tensor):
                predicted_3d_pos = predicted_3d_pos.cpu().numpy()
            results_all.append(predicted_3d_pos)

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    results_all = results_all[:wild_dataset.total_frame()]

    # if visualize, its range will in [-1,1] if needed
    if cfg.detector_3d.estimation_result.export_mp4:
        print(f'>>> Export video result saving...')
        from inference.tool.vismo import render_and_save
        save_path = osp.join('output_3dhpe', osp.basename(cfg.video_path).split('.')[0] + '_3dhpe.mp4')
        render_and_save(results_all, save_path=save_path, keep_imgs=True, fps=vid_fps)
        print(f'>>> Sucessfully save video path in {save_path}')

        if cfg.detector_3d.estimation_result.export_gif:
            print(f'>>> Export gif result saving...')
            from inference.tool.vismo import convert_video2gif
            convert_video2gif(save_path)
            print(f'>>> Sucessfully save video path in {save_path}')

    # if need normalization from pixel range, it will normalized by [0,W] and [0,H] respectively
    # convert to pixel corrdinate,corresponding range is [0,H] and [0,W] respectively
    if cfg.detector_3d.use_pixel_restore:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0

    prediction_world_3d = results_all
    return prediction_world_3d


# transfer camera sapce -> world sapce and generate 3D animation
def mocap_animation(cfg, predicted_camera_3d, vid_fps):
    total_frame = predicted_camera_3d.shape[0]

    from inference.tool.camera import camera_to_world
    # camera rotation matrix in quaternion format
    print('>> Transfer world to world space...')
    rot = np.array(cfg.mocap.camera.extrinsics.R, dtype=np.float32)

    predicted_world_3d = camera_to_world(predicted_camera_3d, R=rot, t=0)

    X = predicted_world_3d[..., 0] * cfg.mocap.scale_factor
    Y = predicted_world_3d[..., 1] * cfg.mocap.scale_factor
    Z = predicted_world_3d[..., 2] * cfg.mocap.scale_factor

    # blender rest pose format: x right y up z forward
    predicted_world_3d[..., 0] = -X
    predicted_world_3d[..., 1] = -Z
    predicted_world_3d[..., 2] = -Y

    print('>> IK solver...')
    # get root trans if needed
    location_sequence = np.zeros([total_frame, 3]) if cfg.mocap.remove_root \
        else predicted_world_3d[..., 0, :]
    # [T,C=3] -> List[np.array(C=3)] and total_length is T
    location_sequence = [item for item in location_sequence]

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

    cur_dir_abs_path = osp.abspath(osp.dirname(__file__))

    tmp_blender_dir = osp.join(cur_dir_abs_path, 'tmp')
    tmp_blender_skeleton_dir = osp.join(tmp_blender_dir, 'skeleton')
    os.makedirs(tmp_blender_dir, exist_ok=True)

    blender_rest_model = osp.join(cur_dir_abs_path, cfg.mocap.animation3d.rest_model_path)
    blender_skeleton_info_script = osp.join(cur_dir_abs_path, cfg.mocap.animation3d.skeleton_constraints_info_path)
    blender_apply_script = osp.join(cur_dir_abs_path, cfg.mocap.animation3d.apply_script_path)

    if os.path.exists(tmp_blender_skeleton_dir):
        shutil.rmtree(tmp_blender_skeleton_dir)

    if not use_bpy:
        os.system(f"blender {blender_rest_model} --background --python {blender_skeleton_info_script}")
    else:
        from inference.tool.blender_skeleton.export_skeleton import external_call_main as export_skeletal_main
        export_skeletal_main(blender_rest_model)

    if not os.path.exists(tmp_blender_skeleton_dir):
        raise Exception("Skeleton export failed")

    from inference.tool.blender_skeleton.ik_solver import SkeletonIKSolver
    ik_solver = SkeletonIKSolver(
        model_path=osp.join(tmp_blender_skeleton_dir),
        frame_rate=vid_fps,
        smooth_range=15 * (1 / vid_fps)
    )

    for t_index in trange(total_frame):
        ik_solver.fit(torch.from_numpy(predicted_world_3d[t_index]).float())

    if cfg.mocap.smooth_eular:
        bone_euler_sequence = ik_solver.get_all_smoothed_bones_euler()
    else:
        bone_euler_sequence = ik_solver.get_all_bones_euler()

    # In our model implementation, rootrel has been considered root transition during train stage
    # location_sequence = ik_solver.get_root_location()
    scale_sequence = ik_solver.get_scale()

    print('>> Saving IK results...')

    if cfg.mocap.bvh.export:
        output_dir_bvh = 'output_bvh'
        os.makedirs(output_dir_bvh, exist_ok=True)
        bvh_file_name = osp.join(output_dir_bvh, osp.basename(cfg.video_path).split('.')[0] + '_ik.bvh')
    else:
        bvh_file_name = None

    import pickle
    export_bvh_format = cfg.mocap.bvh.coordinate_format
    with open(osp.join(tmp_blender_dir, 'bone_animation_data.pkl'), 'wb') as fp:
        FOV = np.pi / 3
        pickle.dump({
            'fov': FOV,
            'frame_rate': vid_fps,
            'bone_names': ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': ik_solver.all_bone_names,
            'bvh_export_path': bvh_file_name,
            'export_bvh_format': export_bvh_format
        }, fp)

    print('>>> Showing 3D animation...')

    if not use_bpy:
        import subprocess
        proc = subprocess.Popen(f"blender {blender_rest_model} --python {blender_apply_script}")
        proc.wait()
    else:
        from inference.tool.blender_skeleton.apply_animation import main as skeletal_animation_main
        skeletal_animation_main()

    # if cfg.mocap.bvh.export and export_bvh_format == 'smart_body':
    #     # add root transition (has problem currently)
    #     # adjust_smart_body_format(bvh_file_name, total_frame,
    #     #                          location_sequence=location_sequence, localtion_scale=(np.mean(scale_sequence),
    #     #                                                                                np.mean(scale_sequence),
    #     #                                                                                np.mean(scale_sequence)))
    #
    #     adjust_motion_format(bvh_file_name, total_frame,
    #                          location_sequence=location_sequence, localtion_scale=(np.mean(scale_sequence),
    #                                                                                     np.mean(scale_sequence),
    #                                                                                     np.mean(scale_sequence)))
    #
    #     # wo root transition
    #     # adjust_smart_body_format(bvh_file_name, total_frame, location_sequence=None)
    # else:
    #     adjust_motion_format(bvh_file_name, total_frame,
    #                               location_sequence=location_sequence, localtion_scale=(np.mean(scale_sequence),
    #                                                                                     np.mean(scale_sequence),
    #                                                                                     np.mean(scale_sequence)))

    # currently, we use bvhio to add root postiion
    adjust_motion_format(bvh_file_name, total_frame,
                         root_name=cfg.mocap.bvh.root_name,
                         location_sequence=location_sequence, localtion_scale=(np.mean(scale_sequence),
                                                                               np.mean(scale_sequence),
                                                                               np.mean(scale_sequence)))

    print(">>> Saved *.bvh file successfully!!")
    print(f">>> Target path is {bvh_file_name}")

    shutil.rmtree(tmp_blender_dir)


# transfer motion sequence into smart body coordinate system
# Here, mainly for rotation operation, the corresponding global offset have some problem currently
# def adjust_smart_body_format(bvh_file_name, frame_num, location_sequence=None, localtion_scale=(1., 1., 1.)):
#     from inference.tool.fairmotion.fairmotion.data import bvh as fairmotion_bvh
#     from inference.tool.fairmotion.fairmotion.core import motion as fairmotion_motion
#     from inference.tool.fairmotion.fairmotion.ops import motion as fairmotion_ops
#     from scipy.spatial.transform.rotation import Rotation as R
#
#     bvh_motion = fairmotion_bvh.load(bvh_file_name)
#     root_ration_matrix = R.from_euler(seq='X', angles=(-90.0)).as_matrix()
#     rotated_bvh_motion = fairmotion_ops.rotate(bvh_motion, R=root_ration_matrix, local=False)
#
#     # add additional offset in bvh
#     if location_sequence is not None:
#         # 1.here, remove init offset at first
#         rotated_bvh_motion_matrix = rotated_bvh_motion.to_matrix()
#         # rotated_bvh_motion_matrix[:, :, 3, :3] -= rotated_bvh_motion_matrix[0:1, :, 3, :3]
#
#         # 2.then, add addtional offset
#         translate_matrix = np.zeros((frame_num, 1, 4, 4))
#         for frame_idx, root_location in enumerate(location_sequence):
#             x, y, z = root_location
#             translate_matrix[frame_idx, 0:1, 3, :3] = np.array(
#                 # [x * localtion_scale[0], -z * localtion_scale[1], y * localtion_scale[2]]
#                 [100, 100, 100]
#             )
#
#         translated_bvh_motion_matrix = rotated_bvh_motion_matrix + translate_matrix
#
#         print('transfers.......................')
#         for frame_idx, root_location in enumerate(location_sequence):
#             print(rotated_bvh_motion_matrix[frame_idx, 0:1, 3, :3], translated_bvh_motion_matrix[frame_idx, 0:1, 3, :3])
#
#         translated_bvh_motion = fairmotion_motion.Motion.from_matrix(translated_bvh_motion_matrix, bvh_motion.skel)
#         bvh_motion.fps = rotated_bvh_motion.fps = translated_bvh_motion.fps = int(vid_fps)
#
#         fairmotion_bvh.save(translated_bvh_motion,
#                             # bvh_file_name,
#                             'verified.bvh',
#                             scale=1.0,
#                             rot_order='ZXY',
#                             verbose=True)
#     # without additional offset in bvh
#     else:
#         fairmotion_bvh.save(rotated_bvh_motion,
#                             bvh_file_name,
#                             scale=1.0,
#                             rot_order='ZXY',
#                             verbose=True)
#
#     print('>>> Trasfer all sequence in smart body coordinate system successfully!!')


# # transfer motion sequence into smart body coordinate system
# # Here, mainly for rotation operation, the corresponding global offset have some problem currently
# def adjust_motion_format(bvh_file_name, frame_num, location_sequence=None, localtion_scale=(1., 1., 1.)):
#     from bvhio import bvhio
#
#     frame_time = bvhio.readAsBvh(bvh_file_name).FrameTime
#     bvh_motion = bvhio.readAsHierarchy(bvh_file_name)
#     root_motion = bvh_motion.filter('Hips')[0]
#
#     # add additional offset in bvh
#     if location_sequence is not None:
#         for frame_idx, root_location in enumerate(location_sequence):
#             root_motion.loadPose(frame_idx, recursive=True)
#             x, y, z = (root_location[0] * localtion_scale[0],
#                        root_location[1] * localtion_scale[1],
#                        root_location[2] * localtion_scale[2])
#             cur_postition = (int(x), int(y), int(z))
#             root_motion.PositionWorld = cur_postition
#             root_motion.Rotation = bvhio.Euler.toQuatFrom((-90, 0, 0), order='ZXY') * root_motion.Rotation
#             root_motion.writePose(frame_idx, recursive=True)
#
#     # without additional offset in bvh
#     else:
#         for frame_idx in range(frame_num):
#             root_motion.loadPose(frame_idx, recursive=True)
#             root_motion.Rotation = bvhio.Euler.toQuatFrom((-90, 0, 0), order='ZXY') * root_motion.Rotation
#             root_motion.writePose(frame_idx, recursive=True)
#
#     bvhio.writeHierarchy(bvh_file_name, root_motion, frame_time)
#     print('>>> Trasfer all sequence in smart body coordinate system successfully!!')


# transfer motion sequence into smart body coordinate system
# Here, mainly for rotation operation, the corresponding global offset have some problem currently
def adjust_motion_format(bvh_file_name, frame_num, root_name: str = 'Hips', location_sequence=None,
                         localtion_scale=(1., 1., 1.)):
    from inference.tool.bvh.modifier.bvhio import bvhio

    frame_time = bvhio.readAsBvh(bvh_file_name).FrameTime
    bvh_motion = bvhio.readAsHierarchy(bvh_file_name)
    root_motion = bvh_motion.filter(root_name)[0]

    root_motion.RestPose.addEuler((90, 0, 0), order='XYZ')
    # root_motion.RestPose.applyPosition((0, 0, 0), recursive=True)
    # root_motion.RestPose.Position = (0, 0, 0)

    # add additional offset in bvh
    if location_sequence is not None:
        for frame_idx, root_location in enumerate(location_sequence):
            root_motion.loadPose(frame_idx, recursive=True)
            x, y, z = (root_location[0] * localtion_scale[0],
                       -root_location[1] * localtion_scale[1],
                       -root_location[2] * localtion_scale[2])
            root_motion.addEuler((-90, 0, 0), order='XYZ')
            cur_postition = (int(x), int(y), int(z))
            # root_motion.applyPosition(cur_postition, recursive=True)
            root_motion.PositionWorld = cur_postition
            root_motion.writePose(frame_idx, recursive=True)

    # without additional offset in bvh
    else:
        for frame_idx in range(frame_num):
            root_motion.loadPose(frame_idx, recursive=True)
            root_motion.addEuler((-90, 0, 0), order='XYZ')
            root_motion.writePose(frame_idx, recursive=True)

    bvhio.writeHierarchy(bvh_file_name, root_motion, frame_time)
    print('>>> Trasfer all sequence in smart body coordinate system successfully!!')


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

    # 2.  2d -> 3d keypoint lifting
    keypoints_2d_combine = np.concatenate([keypoints_2d, keypoints_2d_score], axis=2)
    prediction_camera_3d = inference_pose3d(cfg=cfg, keypoints_2d=keypoints_2d_combine,
                                            vid_w=vid_w, vid_h=vid_h, vid_fps=vid_fps)
    print('>> Successfully predicted 3D keypoints!!')

    # 3. Animation 3D generate
    mocap_animation(cfg, prediction_camera_3d, vid_fps)


if __name__ == '__main__':
    # load config from certain *.yaml config file and cmd
    cfg = combine_cfg_and_args()
    video_file_path = cfg.video_path
    _, vid_fps, vid_size = get_vid_info(video_file_path)
    main(cfg, vid_size, vid_fps)
