from typing import Tuple, List
import numpy as np
import sys
import os.path as osp

sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))


def normalize_incidence_matrix(im: np.ndarray) -> np.ndarray:
    # 这里选择用 [num_vertex,num_vertex] 度矩阵做 D-1*X 的归一化
    Dl = im.sum(-1)
    num_node = im.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    res = Dn @ im
    return res


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    '''
    对出度点和入度点分别构造超图矩阵
    :param num_nodes:
    :param edges:
    :return: 出度和入度的超图矩阵 [num_vertexs,num_edges]
    '''
    source_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    target_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    source_graph = normalize_incidence_matrix(source_graph)
    target_graph = normalize_incidence_matrix(target_graph)
    # print(source_graph.shape,target_graph.shape)
    return source_graph, target_graph


class DiGraph():
    def __init__(self, skeleton):
        '''
        构建有向图，这块建议去看一下HyperFormer中对超图矩阵的定义
        :param skeleton: 由 dataset/lib/skeleton.py 中构建的skeleton骨骼关系连接对象
        # 在Human3.6M中，所有的子节点为 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        #               所有的父节点为 [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        '''
        super().__init__()

        self.num_nodes = len(skeleton.parents())
        # [(parent1,child1),(parent2,child2),...,(parentn,childn)]
        self.directed_edges_hop1 = [(parrent, child) for child, parrent in enumerate(skeleton.parents()) if
                                    parrent >= 0]
        # 下面这三组是手动设计的，是为了消融实验的?
        # 答：其实联系后面的变量名 hopX,可以发现这么一个结论，
        # 这里的hopX 表示一个父结点对应 X 个孩子节点
        # 比如hop2 为 [(parent1,child1_1,child1_2),...,(parentn,childn_1,childn_2)]
        #    hop3 为 [(parrent1,child1,child2,child3),...,(parentn,child1,child2,childn)]
        # 这个hop2在后面的 HighOrder 中用到了
        self.directed_edges_hop2 = [(0, 1, 2), (0, 4, 5), (0, 7, 8), (1, 2, 3), (4, 5, 6), (7, 8, 9), (7, 8, 11),
                                    (7, 8, 14), (8, 9, 10), (8, 11, 12), (8, 14, 15), (11, 12, 13),
                                    (14, 15, 16)]  # (parrent, child)
        self.directed_edges_hop3 = [(0, 1, 2, 3), (0, 4, 5, 6), (0, 7, 8, 9), (7, 8, 9, 10), (7, 8, 11, 12),
                                    (7, 8, 14, 15), (8, 11, 12, 13), (8, 14, 15, 16)]
        self.directed_edges_hop4 = [(0, 7, 8, 9, 10), (0, 7, 8, 11, 12), (0, 7, 8, 14, 15), (7, 8, 11, 12, 13),
                                    (7, 8, 14, 15, 16)]
        # 非超边骨骼数
        self.num_edges = len(self.directed_edges_hop1)
        # 在Human3.6M中,左右两条骨骼边的序号
        self.edge_left = [0, 1, 2, 10, 11, 12]
        self.edge_right = [3, 4, 5, 13, 14, 15]
        self.edge_middle = [6, 7, 8, 9]
        self.center = 0  # for h36m data skeleton
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.directed_edges_hop1)


if __name__ == "__main__":
    from graph.skeleton import get_skeleton_h36m as get_skeleton
    # from graph.skeleton import get_skeleton_mpi_inf_3dhp as get_skeleton

    # 获得skeleton对象
    skeleton = get_skeleton()
    graph = DiGraph(skeleton)
    source_M = graph.source_M
    target_M = graph.target_M
    print(source_M)
    print(target_M)
