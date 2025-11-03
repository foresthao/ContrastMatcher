'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
'''
Collect ego_graph from reduced_oprem.txt for each dataset and store them
'''
import argparse
import torch
from tqdm import tqdm
import os
import sys
import gc
import signal
import random
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from opr.rebuilder import *


def collect_process_nodes_ego(G, k):
    '''
    Collect ego_graph
    '''
    process_nodes_data = {}
    for node in tqdm(G.nodes(data=True), desc='Processing nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            subgraph = nx.ego_graph(G, process_nid, radius=k)
            process_nodes_data[process_nid] = subgraph
    return process_nodes_data

def collect_process_nodes_ego_10(G, k):
    '''
    Collect ego_graph and remove graphs with degree less than 10
    '''
    process_nodes_data = {}
    for node in tqdm(G.nodes(data=True), desc='Collect ego and remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            subgraph = nx.ego_graph(G, process_nid, radius=k)

            # Check if subgraph has more than 10 nodes, if yes then store in process_nodes_data
            if len(subgraph.nodes) > 10:
                process_nodes_data[process_nid] = subgraph
            # print('Check happy')

    return process_nodes_data


def collect_process_nodes_ego_10_batch_pt(G, k, file_path, batch_size=500):#batch_size=1000
    """
    Collect ego_graph, remove graphs with degree less than 10, process in batches and save to multiple .pt files, one file per batch, clear memory after saving
    :param G: Graph object
    :param k: Radius of ego_graph
    :param file_path: Base path for saving files (without filename suffix and batch number)
    :param batch_size: Number of nodes to process per batch, can be adjusted as needed
    """
    batch_count = 0
    file_index = 0
    batch_data = {}  # Temporarily store data for each batch
    for node in tqdm(G.nodes(data=True), desc='Remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            subgraph = nx.ego_graph(G, process_nid, radius=k)
            if len(subgraph.nodes) > 10:
                batch_data[process_nid] = subgraph
                batch_count += 1
                if batch_count % batch_size == 0:
                    file_name = f"{file_path}_batch_{file_index}.pt"
                    try:
                        torch.save(batch_data, file_name)
                    except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
                    # When batch size is reached, clear saved data, flush file buffer and clear memory
                    batch_data = {}
                    gc.collect()
                    file_index += 1
    if batch_data:  # Process last batch (less than batch_size) data saving
        file_name = f"{file_path}_batch_{file_index}.pt"
        try:
            torch.save(batch_data, file_name)
        except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
    print(f"Processed {batch_count} qualified nodes, data saved in batches to multiple .pt files with base path {file_path}.")

def collect_process_nodes_ego_10_batch_pt1(G, k, file_path, batch_size=100):
    """
    Collect ego_graph, remove graphs with degree less than 10, process in batches and save to multiple .pt files, one file per batch, clear memory after saving
    :param G: Graph object
    :param k: Radius of ego_graph
    :param file_path: Base path for saving files (without filename suffix and batch number)
    :param batch_size: Number of nodes to process per batch, can be adjusted as needed
    """
    batch_count = 0
    file_index = 0
    batch_data = {}  # Temporarily store data for each batch
    for node in tqdm(G.nodes(data=True), desc='Remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            subgraph = nx.ego_graph(G, process_nid, radius=k)
            # if len(subgraph.nodes) > 10:
            if subgraph.number_of_nodes() >10:
                batch_data[process_nid] = subgraph
                del subgraph  # Manually delete subgraph to help release memory
                batch_count += 1
                if batch_count % batch_size == 0:
                    file_name = f"{file_path}_batch_{file_index}.pt"
                    try:
                        torch.save(batch_data, file_name)
                    except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
                    # When batch size is reached, clear saved data, flush file buffer and clear memory
                    batch_data = {}
                    gc.collect()
                    file_index += 1
    if batch_data:  # Process last batch (less than batch_size) data saving
        file_name = f"{file_path}_batch_{file_index}.pt"
        try:
            torch.save(batch_data, file_name)
        except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
    print(f"Processed {batch_count} qualified nodes, data saved in batches to multiple .pt files with base path {file_path}.")

def handler(signum, frame):
    raise TimeoutError("Operation timeout, skip current node processing")


def collect_process_nodes_ego_10_batch_pt2(G, k, file_path, batch_size=100):
    """
    Collect ego_graph, remove graphs with degree less than 10, process in batches and save to multiple .pt files, one file per batch, clear memory after saving
    :param G: Graph object
    :param k: Radius of ego_graph
    :param file_path: Base path for saving files (without filename suffix and batch number)
    :param batch_size: Number of nodes to process per batch, can be adjusted as needed
    """
    batch_count = 0
    file_index = 0
    batch_data = {}  # Temporarily store data for each batch
    signal.signal(signal.SIGALRM, handler)  # Set signal handler for timeout processing

    for node in tqdm(G.nodes(data=True), desc='Remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            try:
                signal.alarm(5)  # Set timeout to 5 seconds, can be adjusted as needed
                subgraph = nx.ego_graph(G, process_nid, radius=k)
                # signal.alarm(0)  # Cancel alarm, if operation completes within timeout then cancel subsequent timeout signal
                if subgraph.number_of_nodes() > 10:
                    batch_data[process_nid] = subgraph
                    del subgraph  # Manually delete subgraph to help release memory
                    batch_count += 1
                    if batch_count % batch_size == 0:
                        file_name = f"tem/{file_path}_batch_{file_index}.pt"
                        try:
                            torch.save(batch_data, file_name)
                        except RuntimeError as e:
                            print(f"Error saving data {file_name}: {e}")
                        # When batch size is reached, clear saved data, flush file buffer and clear memory
                        batch_data = {}
                        gc.collect()
                        file_index += 1
                signal.alarm(0)  # Cancel alarm, if operation completes within timeout then cancel subsequent timeout signal
            except TimeoutError as e:
                print(e)
                continue  # If timeout, skip current node and continue to next node

    if batch_data:  # Process last batch (less than batch_size) data saving
        file_name = f"{file_path}_batch_{file_index}.pt"
        try:
            torch.save(batch_data, file_name)
        except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
    print(f"Processed {batch_count} qualified nodes, data saved in batches to multiple .pt files with base path {file_path}.")


def collect_process_nodes_ego_10_batch_pt3(G, k, file_path, batch_size=100):
    """
    Collect ego_graph with 5s skip setting
    """
    batch_count = 0
    file_index = 0
    batch_data = {}  # Temporarily store data for each batch
    signal.signal(signal.SIGALRM, handler)  # Set signal handler for timeout processing

    for node in tqdm(G.nodes(data=True), desc='Remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            try:
                signal.alarm(5)  # Set timeout to 5 seconds, can be adjusted as needed
                subgraph = nx.ego_graph(G, process_nid, radius=k)
                if subgraph.number_of_nodes() > 10:
                    batch_data[process_nid] = subgraph
                    del subgraph  # Manually delete subgraph to help release memory
                    batch_count += 1
                signal.alarm(0)  # Cancel alarm, if operation completes within timeout then cancel subsequent timeout signal
            except TimeoutError as e:
                print(e)
                continue  # If timeout, skip current node and continue to next node

            # Check here if batch size is reached and save data
            if batch_count % batch_size == 0:
                file_name = f"tem/{file_path}_batch_{file_index}.pt"
                try:
                    torch.save(batch_data, file_name)
                except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
                # When batch size is reached, clear saved data, flush file buffer and clear memory
                batch_data = {}
                gc.collect()
                file_index += 1

    if batch_data:  # Process last batch (less than batch_size) data saving
        file_name = f"tem/{file_path}_batch_{file_index}.pt"
        try:
            torch.save(batch_data, file_name)
        except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
    print(f"Processed {batch_count} qualified nodes, data saved in batches to multiple .pt files with base path tem/{file_path}.")


def collect_process_nodes_ego_10_batch_pt4(G, k, file_path, batch_size=100):
    """
    Collect ego_graph, remove graphs with degree less than 10, process in batches and save to multiple .pt files, one file per batch, clear memory after saving, use only 30%
    """
    node_list = list(G.nodes(data=True))  # Convert nodes to list for easier operations
    sample_size = int(len(node_list) * 0.3)  # Calculate number of nodes to sample (rounded)
    sampled_nodes = random.sample(node_list, sample_size)  # Randomly sample 30% of nodes

    batch_count = 0
    file_index = 0
    batch_data = {}  # Temporarily store data for each batch
    signal.signal(signal.SIGALRM, handler)  # Set signal handler for timeout processing

    for node in tqdm(sampled_nodes, desc='Remove graphs with less than 10 nodes'):
        if node[1]['type'] == 'Process':
            process_nid = node[0]
            try:
                signal.alarm(5)  # Set timeout to 5 seconds, can be adjusted as needed
                subgraph = nx.ego_graph(G, process_nid, radius=k)
                if subgraph.number_of_nodes() > 10:
                    batch_data[process_nid] = subgraph
                    del subgraph  # Manually delete subgraph to help release memory
                    batch_count += 1
                signal.alarm(0)  # Cancel alarm, if operation completes within timeout then cancel subsequent timeout signal
            except TimeoutError as e:
                print(e)
                continue  # If timeout, skip current node and continue to next node

            # Check here if batch size is reached and save data
            if batch_count % batch_size == 0:
                file_name = f"tem/{file_path}_batch_{file_index}.pt"
                try:
                    torch.save(batch_data, file_name)
                except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
                # When batch size is reached, clear saved data, flush file buffer and clear memory
                batch_data = {}
                gc.collect()
                file_index += 1

    if batch_data:  # Process last batch (less than batch_size) data saving
        file_name = f"tem/{file_path}_batch_{file_index}.pt"
        try:
            torch.save(batch_data, file_name)
        except RuntimeError as e:
                        print(f"Error saving data {file_name}: {e}")
    print(f"Processed {batch_count} qualified nodes, data saved in batches to multiple .pt files with base path tem/{file_path}.")



def gen_geo_graphs(file, name, k):
    # Create entire graph
    ego_whole = convert_to_networkx(file)
    print('-----------')
    print('Number of nodes:', ego_whole.number_of_nodes())
    print('Number of edges:', ego_whole.number_of_edges())

    # Remove self-loops
    G_without_self_loops = remove_self_loops(ego_whole)
    print('------------')
    print("Number of nodes after removing self-loops:", G_without_self_loops.number_of_nodes())
    print("Number of edges after removing self-loops:", G_without_self_loops.number_of_edges())

    # Remove isolated nodes
    dpg_graph = remove_low_degree_nodes(G_without_self_loops, 1)  # Ran for a long time with no result
    print('-----------')
    print("Number of nodes after removing isolated nodes:", dpg_graph.number_of_nodes())
    print("Number of edges after removing isolated nodes:", dpg_graph.number_of_edges())

    # ego_graphs = collect_process_nodes_ego(G = dpg_graph, k=k)
    # ego_graphs = collect_process_nodes_ego_10(G=dpg_graph, k=k)
    collect_process_nodes_ego_10_batch_pt4(G=dpg_graph, k=k, file_path=name)


if __name__ == '__main__':
    file_trace = './data/DARPA_Engagement3/preduction_reduced/trace/pccache_1000_dpg_1000_threshold_5/ta1-trace-e3-official-1/reduced_oprem.txt'
    file_theia = './data/DARPA_Engagement3/preduction_reduced/theia/pccache_1000_dpg_1000_threshold_5/ta1-theia-e3-official-1r/reduced_oprem.txt'
    file_cadets = './data/DARPA_Engagement3/preduction_reduced/cadets/pccache_1000_dpg_1000_threshold_5/ta1-cadets-e3-official/reduced_oprem.txt'
    # name = 'ego_graph_theia_3hop.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='theia')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    if args.dataset == 'trace':
        file = file_trace
    elif args.dataset == 'theia':
        file = file_theia
    elif args.dataset == 'cadets':
        file = file_cadets
    name = 'ego_graph_' + args.dataset + '_' + str(args.k) +'hot.pt'
    gen_geo_graphs(file=file, name=name, k=args.k)