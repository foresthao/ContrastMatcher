'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
'''
Used to build test set
'''
from torch_geometric.data import Data, Dataset
import torch_geometric.utils as utils
from torch_geometric.loader import DataLoader
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle as pc
import random
import signal
import multiprocessing


# Used to create validation dataset or supervised dataset
class GraphMatchDataset(object):
    def __init__(self, args, dataset, aug = True):
        self.dataset = dataset
        self.aug = aug
    
    def _get_graph(self, idx=None):
        'Get the idx-th graph in the dataset'
        # Randomly select idx
        if idx is None:
            idx = torch.randint(len(self.dataset), (1,)).item()
        return self.dataset[idx]

    def _get_pair(self):
        'Use one of four augmentation methods to augment graph, get a pair of graphs, call _get_graph() method'
        g = self._get_graph()
        if self.aug == True:
            aug_g = subgraph(g)  # Temporarily use subgraph method for sampling, can also be drop_nodes, permute_edges, mask_nodes_fea
            # aug_g = drop_nodes(g)
            # aug_g = g  # Directly no augmentation
        else:
            aug_g = g
        return g, aug_g

    def pairs(self, num_graphs):
        'Generate a pair of graphs and a label'
        # while True:
        batch_graphs = []
        batch_labels = []
        for _ in tqdm(range(num_graphs), desc='Sampling positive samples'):
            g1, g2 = self._get_pair()
            batch_graphs.append((g1, g2))
            batch_labels.append(1)
        # packed_graphs = self._pack_batch(batch_graphs)
        packed_graphs = batch_graphs  # Already using pyg.data type, no need to use this method anymore
        labels = np.array(batch_labels, dtype=np.int32)
        return packed_graphs, labels  # Previously used yield
    
    def neg_pairs(self, num_graphs):  # Only check if isomorphic, easy to get stuck
        # Generate a pair of non-matching graphs and a label
        batch_neg_graphs = []
        batch_neg_labels = []
        for _ in tqdm(range(num_graphs), desc='Sampling negative samples'):
            target = self._get_graph()
            neg_target = self._get_graph()
            neg_q = subgraph(neg_target)
            # This check is too slow, I want to optimize it, remove it before finding optimization method
            target_nx = utils.to_networkx(target)
            # neg_q_nx = utils.to_networkx(neg_q)
            neg_target_nx = utils.to_networkx(neg_target)
            matcher = nx.algorithms.isomorphism.GraphMatcher(target_nx, neg_target_nx)  # Previously was neg_q_nx
            if matcher.subgraph_is_isomorphic():
                continue
            batch_neg_graphs.append((target, neg_target))  # Previously was neg_q
            batch_neg_labels.append(0)
        return batch_neg_graphs, batch_neg_labels
    
    def neg_pairs0(self, num_graphs):
        # No check performed
        batch_neg_graphs = []
        batch_neg_labels = []
        for _ in tqdm(range(num_graphs), desc='Sampling negative samples'):
            target = self._get_graph()
            neg_target = self._get_graph()
            batch_neg_graphs.append((target, neg_target))
            batch_neg_labels.append(0)
        return batch_neg_graphs, batch_neg_labels


    def pack_pair(self, pack_size):
        'Merge return values of pairs and neg_pairs'
        pos_graphs,pos_labels = self.pairs(num_graphs=pack_size)
        neg_graphs,neg_labels = self.neg_pairs0(num_graphs=pack_size)
        graphs = pos_graphs + neg_graphs
        # labels = np.concatenate((pos_labels + neg_labels))
        labels = [1]*len(pos_graphs) + [0]*len(neg_graphs)
        return graphs, labels
    

def drop_nodes(data):
    '''Randomly drop 10% of nodes'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data

def subgraph(data):
    '''Randomly select 20% of nodes to form subgraph'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if n not in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in range(len(idx_nondrop))}

    adj = torch.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    # Create copy of data and modify
    new_data = deepcopy(data)
    new_data.edge_index = edge_index

    # Convert node indices to indices relative to subgraph
    new_data.edge_index[0] = torch.tensor([idx_dict.get(n.item(), -1) for n in new_data.edge_index[0]])
    new_data.edge_index[1] = torch.tensor([idx_dict.get(n.item(), -1) for n in new_data.edge_index[1]])

    # Remove invalid edge indices
    mask = new_data.edge_index[0] != -1
    new_data.edge_index = new_data.edge_index[:, mask]

    # Update node features and labels
    new_data.x = new_data.x[idx_nondrop]
    new_data.y = new_data.y

    return new_data