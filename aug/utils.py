'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import networkx as nx
import GCL.augmentors as A
import torch
import numpy as np
# from .RW import *
from .aug import *

def get_aug_p_old(data, aug_p, pn):  # Initial content, unchanged
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(3, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return A.EdgeRemoving(pe=pn)  # GCL library has issues, may show dropout_adj warning
    elif n == 2:
        return A.RWSampling(num_seeds=num_seeds, walk_length=10)
    else:
        NameError('Augmentation is wrong.')

def get_aug_p(data, aug_p, pn):
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(3, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return EdgeRemoving(pe=pn)
    elif n == 2:
        return RWSampling(num_seeds=num_seeds, walk_length=10)
    else:
        NameError('Augmentation is wrong.')

def get_aug_p_random(data, aug_p, pn):
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(3, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return EdgeRemoving(pe=pn)
    elif n == 2:
        return FeatureMasking(mask_prob=pn)
    else:
        NameError('Augmentation is wrong.')

def get_aug_p_subgraph(data, aug_p, pn):
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(3, 1, p=aug_p)[0]
    if n == 3:#Random walk
        return RWSampling(num_seeds=num_seeds, walk_length=10)
    elif n == 4:#k-core
        return KCoreSubgraph(k=3)
    elif n == 5:#k-truss
        return KTrussSubgraph(k=3)
    else:
        NameError('Augmentation is wrong.')

def get_aug_p6(data, aug_p, pn):
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(6, 1, p=aug_p)[0]
    if n == 0:
        return NodeDropping(pn=pn)
    elif n == 1:
        return EdgeRemoving(pe=pn)
    elif n == 2:#feature mask
        return FeatureMasking(mask_prob=pn)
    elif n == 3:#Random walk
        return RWSampling(num_seeds=num_seeds, walk_length=10)
        # return NodeDropping(pn=pn)
    elif n == 4:#k-core
        return KCoreSubgraph(k=3)
    elif n == 5:#k-truss
        return KTrussSubgraph(k=3)
    else:
        NameError('Augmentation is wrong.')

def get_aug_p2(aug_p, pn):
    # num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(2, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return A.EdgeRemoving(pe=pn)
    else:
        NameError('Augmentation is wrong.')

def mix_aug(data, pn, dataloader, args):
    n = np.random.choice(5,1)[0]
    if n == 0:
        return drop_edge(data, pn)
    elif n == 1:
        return drop_node(data, pn)
    elif n == 2:
        return mask_nodes(data, pn)
    elif n == 3:
        return bfs_subgraph(data, sub_ratio=pn)
    elif n == 4:
        getcore = GetCore(dataloader, 'kcore', pn, args.factor, args.device)
        aug_p=[0.5, 0.5]
        return getcore.drop_joao(data, aug_p)
    else:
        NameError('Augmentation is wrong.')


def drop_edge(data, pn):
    keep_prob = 1. - pn
    probs = torch.tensor([keep_prob for _ in range(data.edge_index.shape[1])]).to(data.x.device)
    edge_mask = torch.bernoulli(probs).to(torch.bool).to(data.x.device)
    edge_index1 = data.edge_index[:, edge_mask]
    return data.x, edge_index1, data.batch



def drop_node(data, pn):
    keep_prob = 1. - pn
    probs = torch.tensor([keep_prob for _ in range(data.x.shape[0])]).to(data.x.device)

    while True:
        node_mask = torch.bernoulli(probs).to(torch.bool).to(data.x.device)
        batch1 = data.batch[node_mask]
        if set(batch1.tolist()) == set(data.batch.tolist()):
            break

    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    edge_index1 = data.edge_index[:, edge_mask]

    # relabel
    x1 = data.x[node_mask]
    node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=data.x.device)
    node_idx[node_mask] = torch.arange(node_mask.sum().item(), device=data.x.device)
    edge_index1 = node_idx[edge_index1]
    return x1, edge_index1, batch1


def mask_nodes(data, pn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * pn)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32, device=device)  # Use normal distribution to replace

    return data.x, data.edge_index, data.batch

def bfs_subgraph(data, sub_ratio):
    """
    Extract a subgraph from the given graph data.

    Args:
        data (torch.data.Data): The PyG data object containing the node features and edge index.
        sub_ratio (float): The ratio of nodes to be kept in the subgraph.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): The node feature tensor, edge index tensor, and batch tensor of the subgraph.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, edge_index, batch = data.x, data.edge_index, data.batch

    if sub_ratio >= 1.0:
        return x, edge_index, batch

    num_nodes = x.size(0)
    sub_num = int(num_nodes * sub_ratio)

    edge_index_np = edge_index.cpu().numpy()

    idx_sub = [np.random.randint(num_nodes, size=1)[0]]
    idx_neigh = set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > num_nodes:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(num_nodes) if n not in idx_sub]
    idx_nondrop = idx_sub

    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index_new = adj.nonzero().t()

    x_new = x[idx_nondrop]  # This is new
    batch_new = batch[idx_nondrop]

    # return x_new, edge_index_new, batch_new
    return x.to(device), edge_index_new.to(device), batch.to(device)


def func1(x):
    return x


def func2(x):
    return torch.sqrt(x)


def func3(x):
    return x * x


class GetCore:
    def __init__(self, dataset, core, pn, frac, device):
        self.func = func3
        self.dataset = dataset
        self.pn = pn
        self.factor = frac
        self.device = device
        self.core = core
        self.weights_dic = self.process_weights()

    def process_weights(self):
        weights_dic = {}

        for data in self.dataset:
            weights = torch.zeros(data.x.shape[0], dtype=torch.float, device=self.device)
            edge_index = data.edge_index.T.cpu().detach().numpy()

            if data.edge_index.shape[1] != 0:
                G = nx.from_edgelist(edge_index)
                core_order = max(nx.core_number(G).values())

                if self.core == 'kcore':
                    for d in [0, 1, 2]:
                        H = nx.k_core(G, max(core_order - d, 1))
                        weights[list(H.nodes)] += 1 / 3

                if self.core == 'ktruss':
                    if G.nodes:
                        k = core_order + 1
                        while k >= 2:
                            H = nx.k_truss(G, k)
                            if H.nodes:
                                break
                            k -= 1
                        truss_order = k
                    else:
                        truss_order = 2

                    for d in [0, 1, 2]:
                        H = nx.k_truss(G, max(truss_order - d, 2))
                        weights[list(H.nodes)] += 1 / 3

            weights_dic[data.idx] = weights

        return weights_dic

    def drop_node(self, data):
        weights = torch.concat([self.weights_dic[i] for i in data.idx.tolist()], dim=0)
        probs = 1. - self.pn * (1 - self.factor * self.func(weights)).to(self.device)

        while True:
            node_mask = torch.bernoulli(probs).to(torch.bool).to(self.device)
            xs = data.x[node_mask]
            batchs = data.batch[node_mask]
            if set(batchs.tolist()) == set(data.batch.tolist()):
                break

        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        edge_indexs = data.edge_index[:, edge_mask]

        # relabel
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=self.device)
        node_idx[node_mask] = torch.arange(node_mask.sum().item(), device=self.device)
        edge_indexs = node_idx[edge_indexs]
        return xs, edge_indexs, batchs

    def drop_edge(self, data):
        weights = torch.concat([self.weights_dic[i] for i in data.idx.tolist()], dim=0)
        probs = 1. - self.pn * (1 - self.factor * self.func(weights)).to(self.device)
        edge_probs = (probs[data.edge_index[0]] + probs[data.edge_index[1]]) / 2
        edge_mask = torch.bernoulli(edge_probs).to(torch.bool)
        edge_indexs = data.edge_index[:, edge_mask]
        return data.x, edge_indexs, data.batch

    def drop_joao(self, data, aug_p):
        n = np.random.choice(2, 1, p=aug_p)[0]
        if n == 0:
            return self.drop_node(data)
        elif n == 1:
            return self.drop_edge(data)
        else:
            NameError('Augmentation is wrong.')

