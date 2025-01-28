import os

import bpy
import pickle

import mathutils
import numpy as np
from mathutils import Matrix
import math


def main():
    INPUT_FILE = 'tmp\\bone_animation_data.pkl'

    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    fov = data['fov']
    frame_rate = data['frame_rate']
    bone_names = data['bone_names']
    bone_euler_sequence = data['bone_euler_sequence']
    location_sequence = data['location_sequence']
    scale = data['scale']
    export_bvh_format = data['export_bvh_format']

    all_bone_names = data['all_bone_names']
    bvh_export_path = data['bvh_export_path']

    root = 'Hips'

    bpy.data.objects['Camera'].location = (0, 0, 0)
    bpy.data.objects['Camera'].rotation_euler = (math.pi / 2., 0, 0)
    # bpy.data.objects['Camera'].rotation_euler = (0, 0, 0)
    bpy.data.objects['Camera'].data.angle = fov

    # set frame rate
    bpy.context.scene.render.fps = int(frame_rate)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(bone_euler_sequence)

    skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
    assert len(skeleton_objs) == 1, "There should be only one skeleton object"
    skeleton = skeleton_objs[0]
    skeleton.location = (0, 0, 0)
    skeleton.rotation_euler = (-math.pi / 2, 0, 0)
    # skeleton.rotation_euler = (0, 0, 0)
    skeleton.scale = (scale, scale, scale)

    # apply animation to skeleton
    for i in range(len(bone_euler_sequence)):
        for j, b in enumerate(bone_names):
            bone = skeleton.pose.bones[b]
            bone.rotation_mode = 'YXZ'
            bone.rotation_euler = bone_euler_sequence[i][j].tolist()
            bone.keyframe_insert(data_path='rotation_euler', frame=i)

        # global location (temporal removed)
        x, y, z = location_sequence[i].tolist()
        skeleton.location = x, z, -y
        skeleton.keyframe_insert(data_path='location', frame=i)

    if bvh_export_path is not None:
        import sys

        sys.path.append(os.path.dirname(__file__))
        from bvh_export import save_custom_bvh

        if export_bvh_format is None:
            # This branch is memrly for simple show in +y up and -z forward bvh viewer,
            # but do not support the rest pose in stand status!!
            global_transfer_matrix = mathutils.Matrix.Rotation(math.radians(180.0), 4, 'X')
        elif export_bvh_format == 'smart_body':
            # Due to blender has different coordinate with default +Y up right hand coordinate system,
            # Here, we need change rest pose(T-pose default here) at first
            # transfer coordinate "Z up y forward" -> "-Z up -y forward"
            global_transfer_matrix = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X') @ \
                                     mathutils.Matrix.Rotation(math.radians(180.0), 4, 'X')
        else:
            assert NotImplementedError, 'Other exported coordinate system do not support currently!!'

        save_custom_bvh(bpy.context, filepath=bvh_export_path,
                        frame_start=1, frame_end=len(bone_euler_sequence),
                        global_scale=1.0,
                        rotate_mode='ZXY',
                        root_transform_only=True,
                        global_matrix=global_transfer_matrix,
                        add_rest_pose_as_first_frame=False)

        print('Trasfer rest pose in smart body coordinate system successfully!!')

        # Then, we use fairmotion to modify pose except rest T-pose in main script: ../../infer_mocap.py
        # please see that script in detail
        print('Back into ../../infer_mocap.py to transfer all motion sequence in smart body coordinate system...')


if __name__ == '__main__':
    main()
