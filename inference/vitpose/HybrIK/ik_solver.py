import os
import os.path as osp
import onnxruntime
from typing import List, DefaultDict, Tuple
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import cv2


class HybrIKEstimator():
    def __init__(self, ckpt_path):
        assert NotImplementedError

    def _pre_process(self, **kwargs):
        assert NotImplementedError

    def predict(self, **kwargs):
        assert NotImplementedError

    def __call__(self, **kwargs):
        self.predict(**kwargs)

    def _post_process(self, **kwargs):
        assert NotImplementedError


class HybrIKEstimatorSinglePersonOnnx(HybrIKEstimator):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)
        assert osp.exists(ckpt_path), "Target checkpoint file doesn't exists!!"
        self.onnx_path = ckpt_path
        self.session = onnxruntime.InferenceSession(ckpt_path)

        self.output_keys_list: List[str] = ['pred_shape', 'pred_theta_mats', 'pred_uvd_jts',
                                            'pred_xyz_jts_29',
                                            'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                                            'pred_xyz_jts_17', 'pred_vertices',
                                            'cam_scale',
                                            'cam_root', 'transl',
                                            'pred_camera']

        self.input_keys_dict: DefaultDict[str, npt.NDArray[np.float32]] = defaultdict()
        self.joint_num = 24

    def _test_transform(self, src: npt.NDArray[np.float32], bbox: List[float]) -> Tuple:
        '''

        Parameters
        ----------
        src [N,H,W,C]
        bbox List[float], length is 4

        Returns [ sliced_image: [C,H=256,W=256], bbox: [4], image_center: [2]  ]
        -------

        '''
        input_size = [256, 256]
        _input_size = input_size
        _aspect_ratio = float(input_size[1]) / input_size[0]
        _scale_mult = 1.25
        _scale_factor = 0.3
        _color_factor = 0.2
        _occlusion = True
        _heatmap_size = [64, 64]
        _depth_dim = 64
        _rot_factor = _rot = 30
        _sigma = 2
        bbox_3d_shape = _bbox_3d_shape = [2.2, 2.2, 2.2]

        # tool transform
        def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
            """Convert box coordinates to center and scale.
            adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
            """
            pixel_std = 1
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5

            if w > aspect_ratio * h:
                h = w / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            scale = np.array(
                [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
            if center[0] != -1:
                scale = scale * scale_mult
            return center, scale

        def get_affine_transform(center,
                                 scale,
                                 rot,
                                 output_size,
                                 shift=np.array([0, 0], dtype=np.float32),
                                 inv=0):

            def get_dir(src_point, rot_rad):
                """Rotate the point by `rot_rad` degree."""
                sn, cs = np.sin(rot_rad), np.cos(rot_rad)

                src_result = [0, 0]
                src_result[0] = src_point[0] * cs - src_point[1] * sn
                src_result[1] = src_point[0] * sn + src_point[1] * cs

                return src_result

            def get_3rd_point(a, b):
                """Return vector c that perpendicular to (a - b)."""
                direct = a - b
                return b + np.array([-direct[1], direct[0]], dtype=np.float32)

            if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
                scale = np.array([scale, scale])

            scale_tmp = scale
            src_w = scale_tmp[0]
            dst_w = output_size[0]
            dst_h = output_size[1]

            rot_rad = np.pi * rot / 180
            src_dir = get_dir([0, src_w * -0.5], rot_rad)
            dst_dir = np.array([0, dst_w * -0.5], np.float32)

            src = np.zeros((3, 2), dtype=np.float32)
            dst = np.zeros((3, 2), dtype=np.float32)
            src[0, :] = center + scale_tmp * shift
            src[1, :] = center + src_dir + scale_tmp * shift
            dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
            dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

            src[2:, :] = get_3rd_point(src[0, :], src[1, :])
            dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

            if inv:
                trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
            else:
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

            return trans

        def _center_scale_to_box(center, scale):
            pixel_std = 1.0
            w = scale[0] * pixel_std
            h = scale[1] * pixel_std
            xmin = center[0] - w * 0.5
            ymin = center[1] - h * 0.5
            xmax = xmin + w
            ymax = ymin + h
            bbox = [xmin, ymin, xmax, ymax]
            return bbox

        def to_tensor(img):
            # (H,W,C) -> (C,H,W)
            img = np.transpose(img, (2, 0, 1))  # C*H*W
            img = img.astype(np.float32)
            if np.max(img) > 1:
                img /= 255
            return img

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, _aspect_ratio, scale_mult=_scale_mult)
        scale = scale * 1.0

        input_size = _input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = to_tensor(img)
        # mean
        mean = np.zeros_like(img)
        mean[0], mean[1], mean[2] = 0.406, 0.457, 0.480
        img[0] -= mean[0]
        img[1] -= mean[1]
        img[2] -= mean[2]

        # std
        std = np.zeros_like(img)
        std[0], std[1], std[2] = 0.225, 0.224, 0.229
        img[0] /= std[0]
        img[1] /= std[1]
        img[2] /= std[2]

        img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox, img_center

    def _pre_process(self, **kwargs) -> Tuple:
        input_img = kwargs['input_image']
        tight_bbox = kwargs['tight_bbox']
        is_rgb_order = kwargs['is_rgb_order']

        if not is_rgb_order:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_img = input_img.astype(np.float32)
        pose_input, bbox, img_center = self._test_transform(input_img, tight_bbox)

        pose_input = np.expand_dims(pose_input, axis=0).astype(np.float32)
        bbox = np.expand_dims(np.array(bbox), axis=0).astype(np.float32)
        img_center = np.expand_dims(np.array(img_center), axis=0).astype(np.float32)

        return pose_input, bbox, img_center

    def __call__(self, *args, **kwargs) -> Tuple:
        return self.predict(**kwargs)

    def predict(self, **kwargs) -> Tuple:
        '''

        Parameters
        ----------
        kwargs
            Dict { input_image: [N,H,W,C] , tight_bbox: List[float],length is 4 , is_rgb_order: True }
        Returns
            pred_rot_vec: [N,3] , pred_shape: [N,10] , transl: [N,3], pred_keypoints17: [N,V*C=17*3]
        -------

        '''
        pose_input, bbox, img_center = self._pre_process(**kwargs)

        self.input_keys_dict['input_image'] = pose_input
        self.input_keys_dict['bbox'] = bbox
        self.input_keys_dict['image_center'] = img_center

        output_tuple = self.session.run(self.output_keys_list, self.input_keys_dict)

        if 'show_warning' in kwargs.keys():
            show_warning = kwargs['show_warning']
        else:
            show_warning = True

        pred_rot_vec, pred_shape, transl, pred_keypoints_17 = self._post_process(output_tuple=output_tuple,
                                                                                 show_warning=show_warning)

        return pred_rot_vec, pred_shape, transl, pred_keypoints_17

    def _post_process(self, show_warning=True, **kwargs):
        def rotmat_to_rotvec(rot_mats):
            from scipy.spatial.transform.rotation import Rotation as R
            assert isinstance(rot_mats, np.ndarray), "only accept numpy for now"
            r = R.from_matrix(rot_mats)
            res = r.as_rotvec()
            return res

        if 'show_warning' in kwargs.keys():
            print(
                '>> Warning: here we only output rotation theta,shape and root translation for standard SMPL currently!!')

        output_tuple = kwargs['output_tuple']
        pred_shape_index = self.output_keys_list.index('pred_shape')
        transl_index = self.output_keys_list.index('transl')
        # transl_index = self.output_keys_list.index('cam_root')
        pred_theta_mats_index = self.output_keys_list.index('pred_theta_mats')
        pred_reg_joint_17_index = self.output_keys_list.index('pred_xyz_jts_17')

        pred_shape, transl, pred_theta_mats, pred_reg_joint_17 = \
            output_tuple[pred_shape_index], output_tuple[transl_index], \
            output_tuple[pred_theta_mats_index], output_tuple[pred_reg_joint_17_index]

        pred_theta_mats = pred_theta_mats.reshape([-1, 3, 3])
        pred_rot_vec = rotmat_to_rotvec(pred_theta_mats).reshape([-1, self.joint_num, 3])

        return pred_rot_vec, pred_shape, transl, pred_reg_joint_17