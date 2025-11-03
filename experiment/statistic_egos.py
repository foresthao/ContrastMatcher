'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import torch
import os
import glob
import networkx as nx

def get_ego_density(dataset_file, h):
    '''
    Read data and calculate average density of ego graphs
    '''
    all_graphs = {}
    if dataset_file == 'data/trace_data/':
        file_pattern = os.path.join(dataset_file, f'ego_graph_trace_{h}hot.pt_batch_*.pt')
    elif dataset_file == 'data/cadets_data/':
        file_pattern = os.path.join(dataset_file, f'ego_graph_cadets_{h}hot.pt_batch_*.pt')
    elif dataset_file == 'data/theia_data/':
        file_pattern = os.path.join(dataset_file, f'ego_graph_theia_{h}hot.pt_batch_*.pt')
    file_paths = glob.glob(file_pattern)
    total_nodes = 0
    total_edges = 0
    num_graphs = 0
    total_density = 0
    for file_path in file_paths:
        # Load stored dictionary from file
        graph_dict = torch.load(file_path)  # Assume using torch to load file, modify according to actual situation
        for graph_id, graph in graph_dict.items():
            all_graphs[graph_id] = graph
            num_graphs += 1
            # Get number of nodes in graph
            num_nodes = graph.number_of_nodes()
            total_nodes += num_nodes
            # Get number of edges in graph
            num_edges = graph.number_of_edges()
            total_edges += num_edges
            # Calculate graph density
            if isinstance(graph, nx.Graph):  # Undirected graph
                density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            elif isinstance(graph, nx.DiGraph):  # Directed graph
                density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            else:
                raise ValueError("Unsupported graph type")
            total_density += density
    average_nodes = total_nodes / num_graphs if num_graphs > 0 else 0
    average_edges = total_edges / num_graphs if num_graphs > 0 else 0
    average_density = total_density / num_graphs if num_graphs > 0 else 0
    return num_graphs, average_nodes, average_edges, average_density


dataset_file = 'data/theia_data/'
num_graphs, average_nodes, average_edges, average_density = get_ego_density(dataset_file, h=3)
print(f"Number of graphs: {num_graphs}")
print(f"Average number of nodes: {average_nodes}")
print(f"Average number of edges: {average_edges}")
print(f"Average density: {average_density}")