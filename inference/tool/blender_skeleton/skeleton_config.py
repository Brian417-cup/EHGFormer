import os
import os.path as osp
import blender_config

REST_POSE_DEFAULT_BONES = [
    'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
    'Hips', 'Spine', 'Spine3', 'Neck',
    'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'
]

# By default, all bones should be optimized here
REST_POSE_OPTIMIZABLE_BONES = [
    'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
    'Hips', 'Spine', 'Spine3',
    'Neck',
    'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'
]

# following standard smartbody human3.6m bvh name style
# This should be same with human3.6m's 17 joints order
HUMAN36M_KEYPOINTS_WITHOUT_HANDS = [
    'Hips',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'Spine', 'Spine3', 'Nose',  # attention: in practice, nose index is missing
    'Neck',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand',
]

TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS = [
    # Head
    ('Spine3', 'Neck'),

    # Body
    ('Hips', 'Spine'),
    ('Spine', 'Spine3'),
    ('Spine3', 'LeftForeArm'),
    ('Spine3', 'RightForeArm'),
    ('Hips', 'RightUpLeg'),
    ('Hips', 'LeftUpLeg'),

    # Arms
    ('LeftForeArm', 'LeftArm'),
    ('LeftHand', 'LeftForeArm'),
    ('RightForeArm', 'RightArm'),
    ('RightHand', 'RightForeArm'),

    # legs
    ('LeftUpLeg', 'LeftLeg'),
    ('RightUpLeg', 'RightLeg'),
    ('LeftLeg', 'LeftFoot'),
    ('RightLeg', 'RightFoot'),

    # followings are constraint bones
    # hands to hips
    ('LeftHand', 'LeftUpLeg'),
    ('RightHand', 'RightUpLeg'),

    # hand to hand
    ('LeftHand', 'RightHand'),
]

##################################################################################
# two optional eular angle constraint
# # in each tuple, the first is min and the second is max
REST_POSE_BONE_CONSTRAINTS = {
    # Euler angles in world space, where human stands z up, -y back
    # followed yxz
    'LeftArm': ((-45, 45), (-60, 75), (-135, 45)),
    'RightArm': ((-45, 45), (-75, 60), (-45, 135)),

    'LeftForeArm': ((-150, 90), (-5, 5), (-135, 5)),
    'RightForeArm': ((-90, 150), (-5, 5), (-5, 135)),

    'LeftHand': ((-5, 5), (-75, 75), (-30, 30)),
    'RightHand': ((-5, 5), (-75, 75), (-30, 30)),

    # 'Spine': ((-5, 15), (-20, 20), (-20, 20)),
    # 'Spine3': ((-5, 15), (-20, 20), (-20, 20)),
    'Spine': ((-30, 30), (-30, 30), (-45, 45)),
    'Spine3': ((-30, 30), (-30, 30), (-45, 45)),

    # 'Head': ((-30, 30), (-30, 30), (-45, 45)),
    'Neck': ((-30, 30), (-30, 30), (-45, 45)),

    'LeftUpLeg': ((-90, 45), (-45, 60), (-45, 45)),
    'RightUpLeg': ((-90, 45), (-60, 45), (-45, 45)),
    'LeftLeg': ((-5, 135), (-15, 15), (-5, 5)),
    'RightLeg': ((-5, 135), (-15, 15), (-5, 5)),
    'LeftFoot': ((-45, 90), (-15, 15), (-45, 45)),
    'RightFoot': ((-45, 90), (-15, 15), (-45, 45)),
}


# another version from
# https://wiki.secondlife.com/wiki/Suggested_BVH_Joint_Rotation_Limits
# REST_POSE_BONE_CONSTRAINTS = {
#     # Euler angles in world space, where human stands z up, -y back
#     # followed yxz
#     'LeftArm': ((-180, 98), (-135, 90), (-91, 97)),
#     'RightArm': ((-98, 180), (-135, 90), (-97, 91)),
#
#     'LeftForeArm': ((-146, 0), (-90, 79), (0, 0)),
#     'RightForeArm': ((0, 146), (-90, 79), (0, 0)),
#
#     'LeftHand': ((-25, 36), (-45, 45), (-90, 86)),
#     'RightHand': ((-36, 25), (-45, 45), (-86, 90)),
#
#     'Spine': ((0, 0), (0, 0), (0, 0)),
#     'Spine3': ((-45, 45), (-45, 22), (-30, 30)),
#
#     # 'Head': ((-30, 30), (-30, 30), (-45, 45)),
#     'Neck': ((-45, 45), (-37, 22), (-30, 30)),
#
#     'LeftUpLeg': ((-85, 105), (-155, 45), (-17, 88)),
#     'RightUpLeg': ((-105, 85), (-155, 45), (-88, 17)),
#     'LeftLeg': ((0, 0), (0, 150), (0, 0)),
#     'RightLeg': ((0, 0), (0, 150), (0, 0)),
#     'LeftFoot': ((-26, 26), (-31, 63), (-74, 15)),
#     'RightFoot': ((-26, 26), (-31, 63), (-15, 74)),
# }

##################################################################################

# another version from
# https://wiki.secondlife.com/wiki/Suggested_BVH_Joint_Rotation_Limits
# REST_POSE_BONE_CONSTRAINTS = {
#     # Euler angles in world space, where human stands z up, -y back
#     # followed yxz
#     'LeftArm': ((-135, 90), (-180, 98), (-91, 97)),
#     'RightArm': ((-135, 90), (-98, 180), (-97, 91)),
#
#     'LeftForeArm': ((-90, 79), (-146, 0), (0, 0)),
#     'RightForeArm': ((-90, 79), (0, 146), (0, 0)),
#
#     'LeftHand': ((-45, 45), (-25, 36), (-90, 86)),
#     'RightHand': ((-45, 45), (-36, 25), (-86, 90)),
#
#     'Spine': ((0, 0), (0, 0), (0, 0)),
#     'Spine3': ((-45, 22), (-45, 45), (-30, 30)),
#
#     # 'Head': ((-30, 30), (-30, 30), (-45, 45)),
#     'Neck': ((-37, 22), (-45, 45), (-30, 30)),
#
#     'LeftUpLeg': ((-155, 45), (-85, 105), (-17, 88)),
#     'RightUpLeg': ((-155, 45), (-105, 85), (-88, 17)),
#     'LeftLeg': ((0, 150), (0, 0), (0, 0)),
#     'RightLeg': ((0, 150), (0, 0), (0, 0)),
#     'LeftFoot': ((-31, 63), (-26, 26), (-74, 15)),
#     'RightFoot': ((-31, 63), (-26, 26), (-15, 74)),
# }

# REST_POSE_BONE_CONSTRAINTS = {
#     # Euler angles in world space, where human stands z up, -y back
#     # followed yxz
#     'LeftArm': ((-135, 90), (91, -97),(-180, 98)),
#     'RightArm': ((-135, 90), (97, -91), (-98, 180)),
#
#     'LeftForeArm': ((-90, 79), (0, 0), (-146, 0)),
#     'RightForeArm': ((-90, 79), (0, 0), (0, 146)),
#
#     'LeftHand': ((-45, 45), (90, -86), (-25, 36)),
#     'RightHand': ((-45, 45), (86, -90), (-36, 25)),
#
#     'Spine': ((0, 0), (0, 0), (0, 0)),
#     'Spine3': ((-45, 22), (30, -30), (-45, 45)),
#
#     # 'Head': ((-30, 30), (45, -45), (-30, 30)),
#     'Neck': ((-37, 22), (30, -30), (-45, 45)),
#
#     'LeftUpLeg': ((-155, 45), (17, -88), (-85, 105)),
#     'RightUpLeg': ((-155, 45), (88, -17), (-105, 85)),
#     'LeftLeg': ((0, 150), (0, 0), (0, 0)),
#     'RightLeg': ((0, 150), (0, 0), (0, 0)),
#     'LeftFoot': ((-31, 63), (74, -15), (-26, 26)),
#     'RightFoot': ((-31, 63), (15, -74), (-26, 26)),
# }

##################################################################################


# For align between 3D pose and rest T-pose model
ALIGN_LOCATION_WITH = ['LeftArm', 'RightArm']
ALIGN_SCALE_WITH = [('LeftArm', 'RightArm'), ('LeftUpLeg', 'RightUpLeg'), ('LeftArm', 'LeftUpLeg'),
                    ('RightArm', 'RightUpLeg')]

if __name__ == '__main__':
    # use extra blender environ

    current_env = os.environ.get('Path')
    new_path = f'{current_env};{blender_config.BLENDER_HOME}'

    os.environ['Path'] = new_path
    blender_rest_model = osp.join(osp.dirname(__file__), '../..', 'rest_model/rest_pose.blend')
    blender_skeleton_info_script = osp.abspath(osp.join(osp.dirname(__file__), 'export_skeleton.py'))
    os.system(f"blender {blender_rest_model} --background --python {blender_skeleton_info_script}")
