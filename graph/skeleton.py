# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)


def get_skeleton_h36m():
    # 这里由于Human3.6M的原始关键点数量是32个，这里减少到常用的17个
    skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                        joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                        joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    # Bring the skeleton to 17 joints instead of the original 32
    skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    # Rewire shoulders to the correct parents
    skeleton._parents[11] = 8
    skeleton._parents[14] = 8
    # Fix children error
    skeleton._children[7] = [8]
    skeleton._children[8] = [9, 11, 14]
    return skeleton


def get_skeleton_h36m_hrnet():
    # 这里由于Human3.6M的原始关键点数量是32个，这里减少到常用的16个(针对HRNnet)
    skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                        joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                        joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    # Bring the skeleton to 16 joints instead of the original 32
    skeleton.remove_joints([4, 5, 9, 10, 11, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    # # Rewire shoulders to the correct parents
    skeleton._parents[8] = 7
    skeleton._parents[10] = 8
    skeleton._parents[13] = 8
    # # Fix children error
    skeleton._children[7] = [8]
    skeleton._children[8] = [9, 10, 13]
    return skeleton


# Inferred from P-STMO And MPI-INF-3DHP
# Attention: The root index is 0 rather than 1
def get_skeleton_mpi_inf_3dhp():
    skeleton = Skeleton(
        parents=[16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
        joints_left=[5, 6, 7, 11, 12, 13], joints_right=[2, 3, 4, 8, 9, 10]
    )
    return skeleton


if __name__ == '__main__':
    # s_h36m = get_skeleton_h36m()
    # # [-1  0  1  2  0  4  5  0  7  8  9  8 11 12  8 14 15]
    # print(s_h36m.parents())
    # print(s_h36m.children())
    # directed_edges = [(parrent, child) for child, parrent in enumerate(s_h36m.parents()) if parrent >= 0]
    # # 打印父结点
    # print([c for p, c in directed_edges])
    # # 打印子节点
    # print([p for p, c in directed_edges])

    # s_3dhp = get_skeleton_mpi_inf_3dhp()
    s_3dhp = get_skeleton_h36m_hrnet()
    print(s_3dhp.parents())
    for i, c in enumerate(s_3dhp.children()):
        for c_i in c:
            print(f'[{i},{c_i}],')
        # print(i, np.array(c))
