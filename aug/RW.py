'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
from torch_sparse import SparseTensor, coalesce
from torch_geometric.utils import subgraph
from collections import deque

class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights

class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()

class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = random_walk_subgraph(edge_index, edge_weights, batch_size=self.num_seeds, length=self.walk_length)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

def random_walk_subgraph_old(edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None, batch_size: int = 1000, length: int = 10):
    num_nodes = edge_index.max().item() + 1

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    start = torch.randint(0, num_nodes, size=(batch_size, ), dtype=torch.long).to(edge_index.device)
    node_idx = adj.random_walk(start.flatten(), length).view(-1)
    unique_node_idx = torch.unique(node_idx)
    edge_index, edge_weight = subgraph(unique_node_idx, edge_index, edge_weight)

    return edge_index, edge_weight


def bfs_walk(start_node, adj_matrix, length):
    visited = set()
    queue = deque([(start_node, 0)])
    nodes = []
    while queue:
        node, depth = queue.popleft()
        if node not in visited:
            visited.add(node)
            nodes.append(node)
            if depth < length:
                neighbors = [i for i, connected in enumerate(adj_matrix[node]) if connected]
                for neighbor in neighbors:
                    queue.append((neighbor, depth + 1))
    return nodes

def random_walk_subgraph(edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None, batch_size: int = 1000, length: int = 10):
    num_nodes = edge_index.max().item() + 1
    row, col = edge_index
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    for i, j in zip(row, col):
        adj_matrix[i.item(), j.item()] = 1

    start = torch.randint(0, num_nodes, size=(batch_size, ), dtype=torch.long).to(edge_index.device)
    all_walks = []
    for s in start:
        walk = bfs_walk(s.item(), adj_matrix, length)
        all_walks.append(torch.tensor(walk))
    node_idx = torch.cat(all_walks).view(-1)
    unique_node_idx = torch.unique(node_idx)
    edge_index, edge_weight = subgraph(unique_node_idx, edge_index, edge_weight)

    return edge_index, edge_weight