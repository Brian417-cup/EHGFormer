import glob
import os
import os.path as osp
import sys

import cv2

sys.path.append(osp.join(osp.dirname(__file__), '../..'))

import numpy as np
from ..nosmpl.vis.vis_o3d import vis_mesh_o3d, Open3DVisualizer
from ..nosmpl.utils import rot_mat_to_euler, rotmat_to_rotvec, quat2mat, quat_to_rotvec
import sys
import pickle
import onnxruntime
import tqdm


class SMPLParameterONNX:
    def __init__(self, smpl_parameter_onnx_path: str) -> None:
        assert osp.exists(smpl_parameter_onnx_path), "Current SMPL parameter for *.onnx doesn't exist!!"
        self.smpl_parameter_onnx_path = smpl_parameter_onnx_path
        self.sess = onnxruntime.InferenceSession(smpl_parameter_onnx_path)

    def forward(self, body_pose_rotvec, global_orient):
        return self.__call__(body_pose_rotvec, global_orient)

    def __call__(self, body_pose_rotvec, global_orient):
        '''
        outputs: vertices, joints, faces
        '''
        outputs = self.sess.run(
            None,
            {
                "global_orient": global_orient,
                "body": body_pose_rotvec,
            },
        )
        return outputs


class SMPLVisualizer():
    def __init__(self, fps: int, frame_width: int, frame_height: int,
                 smpl_parameter_path: str, export_mp4=False, export_gif=False, tmp_img_dir: str = '.',
                 render_video_path: str = ''):
        self.visualizer = Open3DVisualizer(fps=fps, save_img_folder=tmp_img_dir, enable_axis=False,
                                           frame_width=frame_width, frame_height=frame_height)
        self.smpl_parameter = SMPLParameterONNX(smpl_parameter_onnx_path=smpl_parameter_path)
        self.tmp_img_dir = tmp_img_dir
        self.export_mp4 = export_mp4
        self.render_video_name = osp.basename(render_video_path)
        self.export_gif = export_gif

    def forward(self, rot_vec, pred_shape, root_transl,
                extrinsics_matrix_list: list = None, intrinsics_matrix_list: list = None):
        N, T = rot_vec.shape[:2]
        assert N == 1, 'Currently, batch size for render only support 1!!'

        print('>> Start rendering')

        for i, (cur_rot_vec, cur_body_shape, cur_root_transl) in tqdm.tqdm(
                enumerate(zip(rot_vec[0], pred_shape[0], root_transl[0])), total=T
        ):
            global_rot_vec = cur_rot_vec[-1].reshape([1, 1, 3]).astype(np.float32)
            body_rot_vec = cur_rot_vec[:-1].reshape([1, -1, 3]).astype(np.float32)
            outputs = self.smpl_parameter.forward(body_pose_rotvec=body_rot_vec, global_orient=global_rot_vec)

            vertices, joints, faces = outputs
            vertices = vertices[0].squeeze().astype(np.float32)
            joints = joints[0].squeeze().astype(np.float32)
            faces = faces.astype(np.int32)

            # trans = [cur_root_transl[0], cur_root_transl[1], 0]
            trans = [cur_root_transl[0], cur_root_transl[1], cur_root_transl[2]]

            self.visualizer.update(vertices=vertices, faces=faces, trans=trans,
                                   R_along_axis=[np.pi, 0, 0], waitKey=-1,
                                   extrinsics_matrix=extrinsics_matrix_list[0]
                                   if extrinsics_matrix_list is not None else None,
                                   intrinsics_matrix=intrinsics_matrix_list[0]
                                   if intrinsics_matrix_list is not None else None,
                                   )

        self.visualizer.release()

        print('>> Render finish!!')

        if self.export_mp4:
            import imageio
            render_res_video_path = osp.join(self.tmp_img_dir, f"{self.render_video_name.split('.')[0]}_render.mp4")
            videowriter = imageio.get_writer(render_res_video_path, fps=self.visualizer.fps)

            print('>> Converting images into *.mp4 file')
            for file_name in tqdm.tqdm(sorted(glob.glob(osp.join(self.tmp_img_dir, '*.png')))):
                src_img = cv2.imread(file_name)
                cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                videowriter.append_data(src_img)

            videowriter.close()

            print('>> Convert to video success!!')

        if self.export_gif:
            import imageio
            print('>> Converting images into *.gif file')
            gif_frame_list = []
            render_res_gif_path = osp.join(self.tmp_img_dir, f"{self.render_video_name.split('.')[0]}_render.gif")

            for file_name in tqdm.tqdm(sorted(glob.glob(osp.join(self.tmp_img_dir, '*.png')))):
                src_img = cv2.imread(file_name)
                cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                gif_frame_list.append(src_img)
            imageio.mimsave(render_res_gif_path, gif_frame_list, 'GIF', duration=1 / self.visualizer.fps * 1000, loop=0)

            print('>> Convert to gif success!!')


# This is a demo use hybrik .pkl results and with nosmpl repository for visualization
def vis_pkl():
    smpl_onnx = 'data/smpl.onnx'
    smpl = SMPLParameterONNX(smpl_onnx)

    data_f = sys.argv[1]
    data = pickle.load(open(data_f, "rb"))
    ks = list(data.keys())
    print(ks)

    o3d_vis = Open3DVisualizer(fps=60, save_img_folder="results/", enable_axis=False)

    frame_len = len(data[ks[0]])
    print("frames: ", frame_len)
    for fi in range(frame_len):
        pose = data['pred_thetas'][fi]
        pose = np.array(pose).reshape(-1, 3, 3)
        print(pose.shape)

        trans = data["transl"][fi]

        pose_rotvec = [rotmat_to_rotvec(p) for p in pose]
        pose_rotvec = np.array(pose_rotvec).reshape(-1, 3)
        print(pose_rotvec.shape)

        # global_orient = pose_rotvec[:, :3].astype(np.float32)
        global_orient = np.array([[0, 0, 0]], dtype=np.float32).reshape([1, 1, 3])
        # global_orient = [[i[0], -i[1], i[2]] for i in global_orient]
        # global_orient = np.array(global_orient).astype(np.float32)

        pose_rotvec_body = np.delete(pose_rotvec, [-1], axis=0).reshape(1, 23, 3)
        body = pose_rotvec_body.astype(np.float32)
        # lhand = np.zeros([1, 45]).astype(np.float32)
        # rhand = np.zeros([1, 45]).astype(np.float32)

        outputs = smpl(body, global_orient)

        vertices, joints, faces = outputs
        vertices = vertices[0].squeeze()
        joints = joints[0].squeeze()

        faces = faces.astype(np.int32)
        # vis_mesh_o3d(vertices, faces)
        # vertices += trans
        # trans = [trans[1], trans[0], trans[2]]
        trans = [trans[0], trans[1], 0]
        print(trans)
        o3d_vis.update(vertices, faces, trans, R_along_axis=[np.pi, 0, 0], waitKey=1)
    o3d_vis.release()


if __name__ == "__main__":
    vis_pkl()
