import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 22 
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2), (12, 11),
                     (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2), (20, 19), (21, 20), (22, 21),
                     (2, 2)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index] 
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, pseudo_joints=0, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.pseudo_joints = pseudo_joints
        self.A = self.get_adjacency_matrix(labeling_mode)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_ensemble':
            A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, 8)
        elif labeling_mode == 'virtual_ensemble':
            A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.pseudo_joints, 8)
        else:
            raise ValueError()
        return A

if __name__ == '__main__':
    g = Graph(labeling_mode='virtual_spatial')
    print()