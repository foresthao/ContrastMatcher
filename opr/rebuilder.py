'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from opr.opreventmodel import OPREventModel
import networkx as nx
from typing import Union

class OPREventModelRebuilder:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file_and_build_oprems(self, num_lines = None):
        oprems_list = []
        count = 0
        with open(self.file_path, 'r') as f:
            if num_lines is None:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 13:
                        u_parts = {
                            'vid': parts[0],
                            'nid': parts[1],
                            'type': parts[2],
                            'content': parts[3],
                            'flag': parts[4]
                        }
                        v_parts = {
                            'vid': parts[5],
                            'nid': parts[6],
                            'type': parts[7],
                            'content': parts[8],
                            'flag': parts[9]
                        }
                        e_parts = {
                            'type': parts[10],
                            'ts': parts[11],
                            'te': parts[12]
                        }
                        oprem = OPREventModel(u=u_parts, v=v_parts, e=e_parts)
                        oprems_list.append(oprem)
            else:
                while count < num_lines:
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) == 13:
                        u_parts = {
                            'vid': parts[0],
                            'nid': parts[1],
                            'type': parts[2],
                            'content': parts[3],
                            'flag': parts[4]
                        }
                        v_parts = {
                            'vid': parts[5],
                            'nid': parts[6],
                            'type': parts[7],
                            'content': parts[8],
                            'flag': parts[9]
                        }
                        e_parts = {
                            'type': parts[10],
                            'ts': parts[11],
                            'te': parts[12]
                        }
                        oprem = OPREventModel(u=u_parts, v=v_parts, e=e_parts)
                        oprems_list.append(oprem)
                    count += 1 
        return oprems_list

def convert_to_networkx(file_path, num_lines = None):
    '''
    Convert oprem graph to networkx graph
    '''
    rebuilder = OPREventModelRebuilder(file_path)
    if num_lines is None:
        oprems_list = rebuilder.read_file_and_build_oprems()
    else:
        oprems_list = rebuilder.read_file_and_build_oprems(num_lines=num_lines)
    G = nx.MultiDiGraph()
    for oprem in oprems_list:
        u = oprem['u']
        v = oprem['v']
        edge = oprem['e']
        u_nid = u['nid']
        v_nid = v['nid']
        edge_type = edge['type']
        if u_nid not in G.nodes:
            G.add_node(u_nid, type=u['type'])
        if v_nid not in G.nodes:
            G.add_node(v_nid, type=v['type'])
        G.add_edge(u_nid, v_nid, key=edge_type, type=edge_type)
    return G

def remove_zero_degree_nodes(G: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph]):
    """Remove nodes with degree 0 from the provided graph."""
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    for node in nodes_to_remove:
        G.remove_node(node)
    return G

def remove_low_degree_nodes_slow(G: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph], degree_threshold):
    """Remove nodes with degree less than or equal to the given threshold and their incident edges."""
    nodes_to_remove = [node for node, degree in G.degree() if degree <= degree_threshold]
    for node in nodes_to_remove:
        # Remove edges connected to this node
        edges_to_remove = [(u, v) for u, v in G.edges() if node in [u, v]]
        G.remove_edges_from(edges_to_remove)
        # Remove node
        G.remove_node(node)
    return G

def remove_low_degree_nodes(G: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph], degree_threshold):
    '''Improved version: not operating each time, mark all first then operate'''
    nodes_to_remove = []
    edges_to_remove = []
    for node, degree in G.degree():
        if degree <= degree_threshold:
            nodes_to_remove.append(node)
            for neighbor in G.neighbors(node):
                edges_to_remove.append((node, neighbor))
    G.remove_nodes_from(nodes_to_remove)
    G.remove_edges_from(edges_to_remove)
    return G

def remove_self_loops(G: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph]):
    """Remove self-loops from the provided graph."""
    self_loops = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(self_loops)
    return G

def collect_process_nodes(file_path, num_lines, k):
    '''
    Collect ego_graph
    '''
    G = convert_to_networkx(file_path,num_lines=num_lines)
    process_nodes_data = {}
    for node in G.nodes(data=True):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            subgraph = nx.ego_graph(G, process_nid, radius=k)
            process_nodes_data[process_nid] = subgraph
    return process_nodes_data

