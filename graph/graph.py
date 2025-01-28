import numpy as np
import os.path as osp
import sys

import torch

sys.path.append(osp.join(osp.dirname(__file__), '..'))


# 这里参考了ST-GCN动作识别的方法来构件图矩阵

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling 表示单一的， [1,V,V] 邻接矩阵中但凡有连接的元素值都为1
        - distance: Distance Partitioning [max_hop,V,V]  将一个目标邻接矩阵中但凡有连接元素，按照不同的距离值分别放在对应的位置并做以下赋值
                    A[hop,i,j]=hop
        - spatial: Spatial Configuration 思路来自 ST-GCN,这样划分三种矩阵:
                    向心矩阵、离心矩阵和自身根结点范围内的矩阵
                    假设 A[i,j]=hop,即两点间是存在一个距离为hop的连接关系,一个比较依据的中心点下标 center
                    那么 若 A[i,center] > A[j,center] => 离心 / 向心
                        若 A[i,center] < A[j,center] => 向心/ 离心
                        若 A[i,center] = A[j,center] => 自身根结点范围内的矩阵
                    这样一来，三个矩阵分别存储了对应表示 A[i,j] 的标准化后的距离hop值
                    最终 返回 [3,V,V] 分别表示 自身根节点范围内的矩阵、向心矩阵和离心矩阵,分别在对应的类别存储距离hop
        - foa: 对uniform策略的升级版， 生成一个 [3,V,V] 的连接矩阵，包含 自连接矩阵+入度矩阵+出度矩阵，但凡有连接关系的元素值都为1
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 skeleton=None,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        # 选择的某一个点的下标作为中心点，用于 spatial 这种结构的中心点，用来分割出向心矩阵，离心矩阵和自范围中心矩阵
        self.center = 0
        assert strategy in ['uniform', 'distance', 'spatial', 'foa', 'spatial_ctr_gcn']
        self.skeleton = skeleton
        self.get_edge(skeleton)
        # 对邻接矩阵A,得到对应的 [V,V] 表示法， 其中A[i,j]=两点的最短路径距离
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, skeleton):
        # edge is a list of [child, parent] paris
        self.num_node = len(skeleton.parents())
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(child, parrent) for child, parrent in enumerate(skeleton.parents())]
        self.self_link = self_link
        self.neighbor_link = neighbor_link
        self.edge = self_link + neighbor_link

    def get_adjacency(self, strategy):
        # 记录有效距离值
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        # 对每一个hop，只取当前的hop，并把他们置为1，也就是说，对所有的邻接矩阵中的点，但凡它们有连接关系，直接 A[i,j]=1
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        # [V,V]    D^-1@A
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        # 筛选出当前hop下的对应所有元素
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        # 在实际操作中，这里选择的是foa这种策略，即和ST-GCN的动作识别一样，产生了三个矩阵 inward,outward,self
        elif strategy == 'foa':
            A = []
            # 这里其实和uniform策略比较类似，生成一个 [V,V] 自连接矩阵，做  A[i,i]=1
            link_mat = edge2mat(self.self_link, self.num_node)
            # 生成一个 [V, V] 入度矩阵，但凡在原邻接矩阵上有联系的，都做 A[j, i] = 1
            In = normalize_digraph(edge2mat(self.neighbor_link, self.num_node))
            # 生成一个 [V, V] 出度矩阵，但凡在原邻接矩阵上有联系的，都做 A[i, j] = 1
            outward = [(j, i) for (i, j) in self.neighbor_link]
            Out = normalize_digraph(edge2mat(outward, self.num_node))
            # 最终矩阵 = 自连接矩阵+入度矩阵+出度矩阵
            A = np.stack((link_mat, In, Out))
            self.A = A
        # 经过实验验证可以证明，CTR-GCN中用的spatial就是这里的foa模式
        elif strategy == 'spatial_ctr_gcn':
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_reverse = [(parrent, child) for child, parrent in enumerate(self.skeleton.parents())]
            self.A = get_spatial_graph(self.num_node, self_link, self.neighbor_link, neighbor_link_reverse)
        else:
            raise ValueError("Do Not Exist This Strategy")


# Acquire from CTR-GCN
def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_hop_distance(num_node, edge, max_hop=1):
    '''

    :param num_node:
    :param edge: [(start1,end1),(start2,end2),...,(startn,endn)]
    :param max_hop: Determine the max hop, then the valid hop ∈ [0 , max_hop]
    :return: [V,V]
    '''
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    # [A^0,A^1,A^2,...,A^n]
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # 做stack堆叠，并筛选在某个距离衡量下所有点度为0的点 -> [valid_hop,V,V]
    arrive_mat = (np.stack(transfer_mat) > 0)
    # 根据统计得到的结果，对A这个邻接矩阵做矫正， A[i,j]表示的是实际两点间的距离
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    '''
    做D^-1度矩阵的归一化操作
    :param A:
    :return:
    '''
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


if __name__ == '__main__':
    sys.path.append(osp.join(osp.dirname(__file__), '..'))
    keypoint_format = ['h36m', 'mpi-inf-3dhp']
    current_format = 'h36m'
    # current_format = 'mpi-inf-3dhp'
    assert current_format in keypoint_format, 'Desired format is not supported'
    if current_format.lower() == 'h36m':
        from skeleton import get_skeleton_h36m

        get_skeleton = get_skeleton_h36m
    elif current_format.lower() == 'mpi-inf-3dhp':
        from skeleton import get_skeleton_mpi_inf_3dhp

        get_skeleton = get_skeleton_mpi_inf_3dhp
    else:
        raise NotImplementedError

    skeleton = get_skeleton()
    # in practive, we choose this partition strategy
    graph1 = Graph(skeleton=skeleton, strategy='foa', max_hop=1, dilation=1)
    graph2 = Graph(skeleton=skeleton, strategy='spatial_ctr_gcn', max_hop=1, dilation=1)
    graph3 = Graph(skeleton=skeleton, strategy='spatial', max_hop=1, dilation=1)
    # in the hyper method3, we decide to use this graph mode
    graph4 = Graph(skeleton=skeleton, strategy='uniform', max_hop=1, dilation=1)
    graph4_A_tensor=torch.from_numpy(graph4.A).float()
    # print(np.unique(graph1.A))
    print(graph4.A)
    print(graph4.A.shape)
    print(graph4_A_tensor)
    # for i in range(3):
    #     print(f'第{i}个子图中两个模式不相等的下标')
    #     for j in range(17):
    #         for k in range(17):
    #             if graph1.A[i, j, k] != graph2.A[i, j, k]:
    #                 print(j, k)

    # indicies = np.where(graph1.A != graph2.A)
    # print(indicies)
    # print(np.isclose(graph1.A, graph2.A))
    print('Done')
