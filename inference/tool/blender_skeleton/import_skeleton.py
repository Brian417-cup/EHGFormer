import itertools
import torch
import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__)))

import json
import numpy as np
import skeleton_config
from typing import List, Dict, Union


def load_skeleton_data(path: str):
    with open(os.path.join(path, 'skeleton.json'), 'r') as f:
        skeleton = json.load(f)
    bone_names = skeleton['bone_names']
    bone_parents = skeleton['bone_parents']
    # [V,4,4]
    bone_matrix_world_rest = np.load(os.path.join(path, skeleton['bone_matrix_world']))
    # [V,4,4]
    bone_matrix = np.load(os.path.join(path, skeleton['bone_matrix_rel']))
    # when initilize,the corresponding value is None
    skeleton_remap = skeleton['bone_remap']
    skeleton_remap = {k: v for k, v in skeleton_remap.items() if v is not None}
    skeleton_remap.update(
        {k: k for k in skeleton_config.REST_POSE_DEFAULT_BONES if k in bone_names})  # update with default remap

    return bone_names, bone_parents, bone_matrix_world_rest, bone_matrix, skeleton_remap


def get_optimization_target(bone_parents: Dict[str, str], skeleton_remap: Dict[str, str], track_hand: bool = False):
    '''

    Args:
        bone_parents: parent of every bone in rest pose bones
        skeleton_remap: model rest pose bones <-> keypoints from 3D pose
        track_hand: default is False, which is not implemented here when it is True

    Returns:

    '''
    # bone_parents: [key=currentName,value=parentName]
    # skeleton_remap: [key=T pose keypoints name,value=model keypoints name]
    # bones to optimize
    optimizable_bones = [skeleton_remap[b] for b in skeleton_config.REST_POSE_OPTIMIZABLE_BONES if b in skeleton_remap]

    # target pairs
    if track_hand:
        assert NotImplementedError
    else:
        kpt_pairs = [(a, b) for a, b in skeleton_config.TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS if
                     a in skeleton_remap and b in skeleton_remap]

    # print('keypoints_pairs',kpt_pairs)

    # Get bone connection relationshiip
    joint_pairs = [(skeleton_remap[a], skeleton_remap[b]) for a, b in kpt_pairs]

    # Find bones that has target bones as children
    # Get evert chain of its corresponding keypoints
    bone_subset = []
    for t in itertools.chain(*joint_pairs):
        bone_chain = [t]
        while bone_parents[t] is not None:
            t = bone_parents[t]
            bone_chain.append(t)
        for b in reversed(bone_chain):
            if b not in bone_subset:
                bone_subset.append(b)

    if track_hand:
        assert NotImplementedError
    else:
        kpt_pairs_id = torch.tensor(
            [(skeleton_config.HUMAN36M_KEYPOINTS_WITHOUT_HANDS.index(a),
              skeleton_config.HUMAN36M_KEYPOINTS_WITHOUT_HANDS.index(b))
             for a, b in
             kpt_pairs], dtype=torch.long)
    joint_pairs_id = torch.tensor([(bone_subset.index(a), bone_subset.index(b)) for a, b in joint_pairs],
                                  dtype=torch.long)
    # bone_subset: Chains for every joints
    # optimizable_bones:
    # kpt_pairs_id: 视频中的关键点对
    # joint_pairs_id: 模型中的关键点对
    return bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id


def get_constraints(bone_names: List[str], bone_matrix_world_rest: np.ndarray, optimizable_bones: List[str],
                    skeleton_remap: Dict[str, str]):
    # Get constraints
    joint_constraints_id = []
    joint_constraints = []
    for k, c in skeleton_config.REST_POSE_BONE_CONSTRAINTS.items():
        if not (k in skeleton_remap and skeleton_remap[k] in optimizable_bones):
            continue
        b = skeleton_remap[k]
        constraint = []
        rest_mat = bone_matrix_world_rest[bone_names.index(b)]

        # Get local -> world axis
        # This has a premise that each column means respective axis change respectively
        for i in range(3):
            world_axis = np.argmax(np.abs(rest_mat[:3, i]))
            constr = c[world_axis]
            if rest_mat[world_axis, i] < 0:
                constr = -constr[1], -constr[0]
            constraint.append(constr)

        joint_constraints_id.append(optimizable_bones.index(b))
        joint_constraints.append(constraint)

    joint_constraints_id = torch.tensor(joint_constraints_id, dtype=torch.long)
    joint_constraints = torch.tensor(joint_constraints, dtype=torch.float32)

    return joint_constraints_id, torch.deg2rad(joint_constraints)


def get_align_location(bone_names: List[str], skeleton_remap: Dict[str, str]):
    align_location_kpts = torch.tensor(
        [skeleton_config.HUMAN36M_KEYPOINTS_WITHOUT_HANDS.index(k) for k in skeleton_config.ALIGN_LOCATION_WITH],
        dtype=torch.long)
    align_location_joints = torch.tensor(
        [bone_names.index(skeleton_remap[k]) for k in skeleton_config.ALIGN_LOCATION_WITH], dtype=torch.long)
    return align_location_kpts, align_location_joints


def get_align_scale(bone_names: List[str], skeleton_remap: Dict[str, str]):
    align_scale_pairs_kpts = torch.tensor([(skeleton_config.HUMAN36M_KEYPOINTS_WITHOUT_HANDS.index(a),
                                            skeleton_config.HUMAN36M_KEYPOINTS_WITHOUT_HANDS.index(b)) for a, b in
                                           skeleton_config.ALIGN_SCALE_WITH], dtype=torch.long)
    align_scale_pairs_joints = torch.tensor(
        [(bone_names.index(skeleton_remap[a]), bone_names.index(skeleton_remap[b])) for a, b in
         skeleton_config.ALIGN_SCALE_WITH],
        dtype=torch.long)
    return align_scale_pairs_kpts, align_scale_pairs_joints
