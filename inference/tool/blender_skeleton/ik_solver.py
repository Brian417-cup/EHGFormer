import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__)))

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, List
from import_skeleton import load_skeleton_data, get_optimization_target, get_constraints, get_align_location, \
    get_align_scale
from skeleton_config import HUMAN36M_KEYPOINTS_WITHOUT_HANDS
from utils3d import euler_angle_to_matrix, mls_smooth


@torch.jit.script
def barrier(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return torch.exp(4 * (x - b)) + torch.exp(4 * (a - x))


def eval_matrix_world(parents: torch.Tensor, matrix_bones: torch.Tensor, matrix_basis: torch.Tensor) -> torch.Tensor:
    "Deprecated"
    matrix_bones, matrix_basis = matrix_bones.unbind(), matrix_basis.unbind()
    matrix_world = []
    for i in range(len(matrix_bones)):
        local_mat = torch.mm(matrix_bones[i], matrix_basis[i])
        m = local_mat if parents[i] < 0 else torch.mm(matrix_world[parents[i]], local_mat)
        matrix_world.append(m)
    return torch.stack(matrix_world)


# A more faster inference by directly using C++
# import ctypes
#
#
# class EvalMatrixWorld(torch.autograd.Function):
#     """
#     Call c++ function to evaluate matrix_world
#     """
#
#     cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'cpp_eval_bone_matrix/cpp_eval_bone_matrix.dll'))
#     cpp_eval_matrix_world = cdll.eval_matrix_world
#     cpp_grad_matrix_world = cdll.grad_matrix_world
#
#     @staticmethod
#     def forward(ctx, parents: torch.Tensor, matrix_bones: torch.Tensor, matrix_basis: torch.Tensor):
#         assert parents.dtype == torch.int64 and parents.is_contiguous()
#         assert matrix_bones.dtype == torch.float32 and matrix_bones.is_contiguous()
#         assert matrix_basis.dtype == torch.float32 and matrix_basis.is_contiguous()
#
#         matrix_world = torch.zeros_like(matrix_bones)
#
#         EvalMatrixWorld.cpp_eval_matrix_world(
#             ctypes.c_ulonglong(parents.shape[0]),
#             ctypes.c_void_p(parents.data_ptr()),
#             ctypes.c_void_p(matrix_bones.data_ptr()),
#             ctypes.c_void_p(matrix_basis.data_ptr()),
#             ctypes.c_void_p(matrix_world.data_ptr()),
#         )
#         ctx.save_for_backward(parents, matrix_bones, matrix_basis, matrix_world)
#         return matrix_world
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         assert grad_out.dtype == torch.float32 and grad_out.is_contiguous()
#
#         parents, matrix_bones, matrix_basis, matrix_world = ctx.saved_tensors
#         grad_matrix_basis = torch.zeros_like(matrix_basis)
#         grad_matrix_world = grad_out.clone()
#         EvalMatrixWorld.cpp_grad_matrix_world(
#             ctypes.c_ulonglong(parents.shape[0]),
#             ctypes.c_void_p(parents.data_ptr()),
#             ctypes.c_void_p(matrix_bones.data_ptr()),
#             ctypes.c_void_p(matrix_basis.data_ptr()),
#             ctypes.c_void_p(matrix_world.data_ptr()),
#             ctypes.c_void_p(grad_matrix_basis.data_ptr()),
#             ctypes.c_void_p(grad_matrix_world.data_ptr()),
#         )
#         return None, None, grad_matrix_basis
#
#
# eval_matrix_world = EvalMatrixWorld.apply


class SkeletonIKSolver(object):
    def __init__(self, model_path: str, frame_rate: float, smooth_range: float = 0.3):
        # load skeleton model data (offered by blender)
        all_bone_names, all_bone_parents, all_bone_matrix_world_rest, all_bone_matrix, skeleton_remap = load_skeleton_data(
            model_path)
        self.all_bone_names: List[str] = all_bone_names
        self.all_bone_parents: List[str] = all_bone_parents
        self.all_bone_parents_id = torch.tensor(
            [(all_bone_names.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in
             all_bone_parents], dtype=torch.long)
        self.all_bone_matrix: torch.Tensor = torch.from_numpy(all_bone_matrix).float()
        self.track_hands = False

        self.h36m_skeleton = HUMAN36M_KEYPOINTS_WITHOUT_HANDS
        # Here, kpt comes from Pose3D, joint from RestPose model
        bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id = get_optimization_target(all_bone_parents,
                                                                                               skeleton_remap,
                                                                                               self.track_hands)

        # pairs indicies of
        self.joint_pairs_a, self.joint_pairs_b = joint_pairs_id[:, 0], joint_pairs_id[:, 1]
        self.kpt_pairs_a, self.kpt_pairs_b = kpt_pairs_id[:, 0], kpt_pairs_id[:, 1]

        self.bone_parents_id = torch.tensor(
            [(bone_subset.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in bone_subset],
            dtype=torch.long)
        subset_id = [all_bone_names.index(b) for b in bone_subset]
        self.bone_matrix = self.all_bone_matrix[subset_id]

        # joint constraints (but from the get_contstraints function, the joint is equal to bone)
        joint_constraint_id, joint_constraint_value = get_constraints(all_bone_names, all_bone_matrix_world_rest,
                                                                      optimizable_bones, skeleton_remap)
        self.joint_contraint_id = joint_constraint_id
        self.joint_constraints_min, self.joint_constraints_max = joint_constraint_value[:, :,
                                                                 0], joint_constraint_value[:, :, 1]

        # align location index(from image 3D pose the left and right joints in symmetry way)
        self.align_location_kpts, self.align_location_bones = get_align_location(optimizable_bones, skeleton_remap)

        # align scale index(the left and right joints in symmetry way)
        self.align_scale_pairs_kpt, self.align_scale_pairs_bone = get_align_scale(all_bone_names, skeleton_remap)
        rest_joints = torch.from_numpy(all_bone_matrix_world_rest)[:, :3, 3]
        self.align_scale_pairs_length = torch.norm(
            rest_joints[self.align_scale_pairs_bone[:, 0]] - rest_joints[self.align_scale_pairs_bone[:, 1]], dim=-1)

        # optimization hyperparameters
        self.lr = 1.0
        self.max_iter = 50
        self.tolerance_change = 1e-6
        self.tolerance_grad = 1e-4
        self.joint_constraint_loss_weight = 1
        self.pose_reg_loss_weight = 0.1
        self.smooth_range = smooth_range

        # optimizable bone euler angles
        self.optimizable_bones = optimizable_bones
        self.gather_id = torch.tensor(
            [(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in bone_subset], dtype=torch.long)[
                         :, None, None].repeat(1, 4, 4)
        self.all_gather_id = torch.tensor(
            [(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in all_bone_names],
            dtype=torch.long)[:, None, None].repeat(1, 4, 4)
        # The optimized matrix is for bone rather than keypoint here
        self.optim_bone_euler = torch.zeros((len(optimizable_bones), 3), requires_grad=True)

        # for optimizer history
        self.euler_angle_history: List[npt.NDArray[np.float32]] = []
        self.location_history: List[npt.NDArray[np.float32]] = []
        self.align_scale_history: List[float] = []
        self.frame_rate = frame_rate

    def fit(self, kpts: Union[torch.tensor, npt.NDArray[np.float32]]):
        optimizer = torch.optim.LBFGS(
            [self.optim_bone_euler],
            line_search_fn='strong_wolfe',
            lr=self.lr,
            max_iter=100 if len(self.euler_angle_history) == 0 else self.max_iter,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad
        )

        # here, we assume all keypoints is all valid
        kpt_pairs_a, kpt_pairs_b = self.kpt_pairs_a, self.kpt_pairs_b
        joint_pairs_a, joint_pairs_b = self.joint_pairs_a, self.joint_pairs_b

        kpt_dir = kpts[kpt_pairs_a] - kpts[kpt_pairs_b]
        kpt_pairs_length = torch.norm(kpts[self.align_scale_pairs_kpt[:, 0]] - kpts[self.align_scale_pairs_kpt[:, 1]],
                                      dim=-1)
        align_scale = (kpt_pairs_length / self.align_scale_pairs_length).mean()
        if align_scale > 0:
            # self.align_scale = align_scale
            kpt_dir = kpt_dir / align_scale

        def _loss_closure():
            optimizer.zero_grad()
            # transform matrix (The order of 'YXZ' is to reduce danger of rotation lock)
            optim_matrix_basis = euler_angle_to_matrix(self.optim_bone_euler, 'YXZ')
            # concat is for the root joint and others
            matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0,
                                        index=self.gather_id)
            matrix_world = eval_matrix_world(self.bone_parents_id, self.bone_matrix, matrix_basis)
            joints = matrix_world[:, :3, 3]
            joint_dir = joints[joint_pairs_a] - joints[joint_pairs_b]
            dir_loss = F.mse_loss(kpt_dir, joint_dir)
            joint_prior_loss = barrier(self.optim_bone_euler[self.joint_contraint_id], self.joint_constraints_min,
                                       self.joint_constraints_max).mean()
            pose_reg_loss = self.optim_bone_euler.square().mean()
            loss = dir_loss + self.pose_reg_loss_weight * pose_reg_loss + self.joint_constraint_loss_weight * joint_prior_loss
            loss.backward()
            return loss

        if len(kpt_dir) > 0:
            optimizer.step(_loss_closure)

        optim_matrix_basis = euler_angle_to_matrix(self.optim_bone_euler, 'YXZ')
        matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0,
                                    index=self.all_gather_id)
        matrix_world = torch.tensor([align_scale, align_scale, align_scale, 1.])[None, :, None] * eval_matrix_world(
            self.bone_parents_id, self.bone_matrix, matrix_basis)
        location = kpts[self.align_location_kpts].mean(dim=0) - matrix_world[self.align_location_bones, :3, 3].mean(
            dim=0)

        self.euler_angle_history.append(self.optim_bone_euler.detach().clone().cpu().numpy())
        self.location_history.append(location.detach().clone().cpu().numpy())
        self.align_scale_history.append(align_scale)

    def get_all_bones_euler(self) -> List[npt.NDArray[np.float32]]:
        return self.euler_angle_history

    def get_all_smoothed_bones_euler(self) -> List[npt.NDArray[np.float32]]:
        bone_eular_pair = [(torch.from_numpy(item), i * 1.0 / self.frame_rate) for i, item in
                           enumerate(self.euler_angle_history)]
        smoothed_bones_eular: List[npt.NDArray[np.float32]] = []

        def _get_smoothed_single_bone_euler(query_t: float) -> npt.NDArray[np.float32]:
            input_euler, input_t = zip(
                *((e, t) for e, t in bone_eular_pair if abs(t - query_t) < self.smooth_range))
            if len(input_t) <= 2:
                joints_smoothed = input_euler[-1]
            else:
                joints_smoothed = mls_smooth(input_t, input_euler, query_t, self.smooth_range).cpu().detach().numpy()
            return joints_smoothed

        query_t = 0.0
        for _ in self.euler_angle_history:
            smoothed_eular = _get_smoothed_single_bone_euler(query_t)
            smoothed_bones_eular.append(smoothed_eular)
            query_t += 1.0 / self.frame_rate

        return smoothed_bones_eular

    def get_root_location(self) -> List[npt.NDArray[np.float32]]:
        return self.location_history

    def get_scale(self) -> List[float]:
        return self.align_scale_history
