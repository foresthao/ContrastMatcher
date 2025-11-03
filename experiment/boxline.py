'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
from dataset_to_pt import convert_to_networkx, remove_self_loops, remove_low_degree_nodes
import os
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def gen_geo_graphs(file_path):
    trace_file = os.path.join(file_path, 'trace/pccache_1000_dpg_1000_threshold_5/ta1-trace-e3-official-1/reduced_oprem.txt')
    cadets_file = os.path.join(file_path, 'cadets/pccache_1000_dpg_1000_threshold_5/ta1-cadets-e3-official/reduced_oprem.txt')
    theia_file = os.path.join(file_path, 'theia/pccache_1000_dpg_1000_threshold_5/ta1-theia-e3-official-1r/reduced_oprem.txt')
    # Process trace dataset
    ego_whole_trace = convert_to_networkx(trace_file)

    print('-----------')
    print('Trace number of nodes:', ego_whole_trace.number_of_nodes())
    print('Trace number of edges:', ego_whole_trace.number_of_edges())

    G_without_self_loops_trace = remove_self_loops(ego_whole_trace)
    print('------------')
    print("Trace number of nodes after removing self-loops:", G_without_self_loops_trace.number_of_nodes())
    print("Trace number of edges after removing self-loops:", G_without_self_loops_trace.number_of_edges())

    dpg_graph_trace = remove_low_degree_nodes(G_without_self_loops_trace, 1)
    print('-----------')
    print("Trace number of nodes after removing isolated nodes:", dpg_graph_trace.number_of_nodes())
    print("Trace number of edges after removing isolated nodes:", dpg_graph_trace.number_of_edges())

    # Process cadets dataset
    ego_whole_cadets = convert_to_networkx(cadets_file)
    print('-----------')
    print('Cadets number of nodes:', ego_whole_cadets.number_of_nodes())
    print('Cadets number of edges:', ego_whole_cadets.number_of_edges())

    G_without_self_loops_cadets = remove_self_loops(ego_whole_cadets)
    print('------------')
    print("Cadets number of nodes after removing self-loops:", G_without_self_loops_cadets.number_of_nodes())
    print("Cadets number of edges after removing self-loops:", G_without_self_loops_cadets.number_of_edges())

    dpg_graph_cadets = remove_low_degree_nodes(G_without_self_loops_cadets, 1)
    print('-----------')
    print("Cadets number of nodes after removing isolated nodes:", dpg_graph_cadets.number_of_nodes())
    print("Cadets number of edges after removing isolated nodes:", dpg_graph_cadets.number_of_edges())

    # Process theia dataset
    ego_whole_theia = convert_to_networkx(theia_file)
    print('-----------')
    print('Theia number of nodes:', ego_whole_theia.number_of_nodes())
    print('Theia number of edges:', ego_whole_theia.number_of_edges())

    G_without_self_loops_theia = remove_self_loops(ego_whole_theia)
    print('------------')
    print("Theia number of nodes after removing self-loops:", G_without_self_loops_theia.number_of_nodes())
    print("Theia number of edges after removing self-loops:", G_without_self_loops_theia.number_of_edges())

    dpg_graph_theia = remove_low_degree_nodes(G_without_self_loops_theia, 1)
    print('-----------')
    print("Theia number of nodes after removing isolated nodes:", dpg_graph_theia.number_of_nodes())
    print("Theia number of edges after removing isolated nodes:", dpg_graph_theia.number_of_edges())

    return [dpg_graph_trace, dpg_graph_cadets, dpg_graph_theia]
    # return [dpg_graph_trace, dpg_graph_cadets]

def random_walk_3_hop(G):
    source_node = random.choice(list(G.nodes()))
    visited = set([source_node])
    current_hop = 0
    current_neighbors = [source_node]
    # start_time = time.time()
    start_time = time.perf_counter()
    while current_hop < 3:
        next_neighbors = []
        for node in current_neighbors:
            neighbors = list(G.neighbors(node))
            next_neighbors.extend(neighbors)
        visited.update(next_neighbors)
        current_neighbors = next_neighbors
        current_hop += 1
    # end_time = time.time()
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    file_path = 'data/DARPA_Engagement3/preduction_reduced'
    # data/DARPA_Engagement3/preduction_reduced
    datasets = gen_geo_graphs(file_path)
    num_experiments = 10000
    time_data = {f'dataset_{i}': [] for i in range(len(datasets))}

    for i, G in enumerate(datasets):
        for _ in range(num_experiments):
            walk_time = random_walk_3_hop(G)
            time_data[f'dataset_{i}'].append(walk_time)

    # Convert data to DataFrame for plotting
    df = pd.DataFrame(time_data)
    df.columns = ['trace', 'cadets', 'theia']

    # Set Chinese font support
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create larger figure
    plt.figure(figsize=(10, 8))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Plot boxplot
    ax = sns.boxplot(data=df, width=0.6, linewidth=2.0)
    
    # Set axis labels and font size
    # plt.xlabel('Datasets', fontsize=20, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=20, fontweight='bold')
    
    # Set tick label font size
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    
    # Set y-axis range and log scale
    ymin = 1e-7
    ymax = 1
    plt.ylim(ymin, ymax)
    plt.yscale('log')
    
    # Beautify chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    # Set grid line style
    ax.grid(True, alpha=0.5, linewidth=1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('3_hop_traversal_time_consumption2.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()