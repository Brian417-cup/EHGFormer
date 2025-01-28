# Attention: This script is used in blender for python, rather than directly used in current python environment!!
import os
import bpy
import numpy as np
import json

SAVE_DIR = "tmp\\skeleton"
os.makedirs(SAVE_DIR, exist_ok=True)


def external_call_main(rest_model_blend_file_path: str = None):
    bpy.ops.wm.open_mainfile(filepath=rest_model_blend_file_path)
    main(use_bpy=True)


def main(use_bpy=False):
    def iterdfs(bone):
        yield bone
        for child in bone.children:
            for descent in iterdfs(child):
                yield descent

    def iterbones(bones):
        for r in filter(lambda b: b.parent is None, bones):
            for b in iterdfs(r):
                yield (b)

    def export_bones(skeleton):
        bones = skeleton.pose.bones
        bone_names = [b.name for b in iterbones(bones)]
        bone_parents = {b: bones[b].parent.name if bones[b].parent is not None else None for b in bone_names}

        bone_matrix_rel, bone_matrix_world = [], []
        for bn in bone_names:
            b = bones[bn]
            bone_matrix_world.append(np.array(skeleton.matrix_world @ b.matrix, dtype=np.float32))
            # as for the local matrix caculation,blender use following attribute:
            # refered from: https://blender.stackexchange.com/questions/35125/what-is-matrix-basis
            # matrix_local = matrix_parent_inverse * matrix_basis
            # matrix_world = parent.matrix_world * matrix_local

            # print(bn, b.matrix_basis)

            if b.parent is None:
                m = np.array(skeleton.matrix_world @ b.matrix @ b.matrix_basis.inverted(), dtype=np.float32)
            else:
                m = np.array(b.parent.matrix.inverted() @ b.matrix @ b.matrix_basis.inverted(), dtype=np.float32)
            bone_matrix_rel.append(m)
        return bone_names, bone_parents, np.stack(bone_matrix_rel), np.stack(bone_matrix_world)

    def export_numpy(info: dict, prefix=[]):
        for k in info:
            prefix_ = prefix + [k]
            if isinstance(info[k], dict):
                export_numpy(info[k], prefix=prefix_)
            elif isinstance(info[k], np.ndarray):
                # make sure the array is C contiguous
                print(k, info[k].dtype, info[k].shape, info[k].flags.c_contiguous)
                filename = str('_').join(prefix_) + '.npy'
                np.save(os.path.join(SAVE_DIR, filename), info[k])
                info[k] = filename

    def save_json_with_numpy(info: dict, file):
        export_numpy(info)
        with open(file, 'w') as f:
            json.dump(info, f, indent=4)

    skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
    assert len(skeleton_objs) == 1, "There should be only one skeleton object"
    bone_names, bone_parents, bone_matrix_rel, bone_matrix_world = export_bones(skeleton_objs[0])

    # Save skeleton
    # the bone_matrix_rel and bone_matrix_worl keys are save merely *.npy path, and the outputs are in *.npy path
    skeleton = {
        'bone_names': bone_names,
        'bone_parents': bone_parents,
        'bone_matrix_rel': bone_matrix_rel,
        'bone_matrix_world': bone_matrix_world,
        'bone_remap': {
            'Hips': None,
            'RightUpLeg': None,
            'RightLeg': None,
            'RightFoot': None,
            "LeftUpLeg": None,
            "LeftLeg": None,
            "LeftFoot": None,
            "Spine": None,
            "Spine3": None,
            "Neck": None,
            "LeftArm": None,
            "LeftForeArm": None,
            "LeftHand": None,
            "RightArm": None,
            "RightForeArm": None,
            "RightHand": None,
        }
    }

    save_json_with_numpy(skeleton, os.path.join(SAVE_DIR, 'skeleton.json'))

    if use_bpy:
        bpy.ops.wm.quit_blender()
    else:
        quit()


if __name__ == '__main__':
    main()
