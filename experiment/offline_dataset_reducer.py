'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import sys
import os
import re
import argparse
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from opr import *
from opr.datasaver import GraphSaver
from opr.reduction import OnlineReduction

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from experiment.utils import timeit
from opr import *

def parser_txt_extract_number(filepath: str) ->int :
    # Used to extract filename
    match = re.search(r'(\d+)_parser_items\.txt', filepath)
    if match:
        return int(match.group(1))
    else:
        return 0

def get_parsed_filepaths(dataset_path: str) -> list:
    filepaths = []
    for file in os.listdir(dataset_path):
        if file.endswith('parser_items.txt'):
            filepaths.append(os.path.join(dataset_path, file))
    filepaths = sorted(filepaths, key = parser_txt_extract_number)
    return filepaths

@timeit
def reducing_subdataset(dataset:str, subdataset:str, save:bool, save_mode:str):
    global reducer_map
    d_reducer_map = reducer_map[dataset]

    subdata_path = os.path.join(d_reducer_map['parsed_path'], subdataset)
    subsave_path = os.path.join(d_reducer_map['save_path'], subdataset)
    oprem_path = os.path.join(subsave_path, 'reduced_oprem.txt')
    template_path = os.path.join(subsave_path, 'templates.json')
    if not os.path.exists(subsave_path):
        os.makedirs(subsave_path)
    parsed_filepaths = get_parsed_filepaths(subdata_path)

    reduction = OnlineReduction()
    if save:
        gs = GraphSaver(save_mode=save_mode, oprem_path=oprem_path)

    @timeit
    def reducing_single_file(filepath: str):
        f = open(filepath, 'r')
        for line in f:
            oprem = OPREventModel()
            oprem.update_from_loprem(line.strip().split('\t'))
            all_node_events = reduction.consume_single_OPREM(oprem)
            for node_events in all_node_events:
                for event in node_events:
                    if save:
                        gs.save_OPREM(event)
        # reduction.remove_degree_zero_nodes()  # Remove nodes with degree 0
        f.close()
    
    # Reduce each file
    for parsed_items_path in parsed_filepaths:
        print(f'===Reducing file: {parsed_items_path}===')
        reducing_single_file(parsed_items_path)
        print('===Reducing file finished===')

    # Move from pcache to dpg
    reduction.pop_pccache_to_dpg()
    all_node_events = reduction.dpg.pop_all_nodes()
    for node_events in all_node_events:
        for event in node_events:
            if save:
                gs.save_OPREM(event)
    
    reduction.templates.save_json(template_path)
    if save:
        gs.close()


def reducing_dataset(dataset:str, save:bool, save_mode:str):
    global reducer_map
    parsed_path = reducer_map[dataset]['parsed_path']
    folders = [name for name in os.listdir(parsed_path) if os.path.isdir(os.path.join(parsed_path,name))]
    for subdataset in folders:
        print(f'\n======================Reducing subdataset {subdataset}======================')
        reducing_subdataset(dataset, subdataset, save, save_mode)

def start_reducting(dataset:str, subdataset:str, save:bool, save_mode:str):
    if 'e3' in dataset:
        if subdataset is None:
            reducing_dataset(dataset, save, save_mode)
        else:
            print(f'\n======================Reducing subdataset {subdataset}======================')
            reducing_subdataset(dataset, subdataset, save, save_mode)
    elif 'e5' in dataset:
        print('\nE5 not processed yet')

if __name__ == '__main__':
    reducer_map = {
        'e3theia': {
            'parsed_path':'./data/sj/DARPA_Engagement3/preduction_parsed/theia',
            'save_path': './data/sj/DARPA_Engagement3/preduction_reduced/theia'
        },
        'e3trace':{
            'parsed_path':'./data/DARPA_Engagement3/preduction_parsed/trace',
            'save_path': './data/DARPA_Engagement3/preduction_reduced/trace'
        },
        'e3cadets':{
            'parsed_path':'./data/sj/DARPA_Engagement3/preduction_parsed/cadets',
            'save_path': './data/sj/DARPA_Engagement3/preduction_reduced/cadets'
        }
    }

    # Set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', default='True', help='if set, save the OPREM event')
    parser.add_argument('--m', '--save_mode', type=str, default='oprem', help='Save mode for the OPREM events')
    parser.add_argument('--dataset', type=str, default='e3trace', help='The dataset to reduce')
    parser.add_argument('--subdataset', type=str, default='ta1-trace-e3-official-1', help='The subdataset to reduce, must be the subdirectory of the dataset')
    parser.add_argument('-p', '--pccache_window', type=int, default=500, help='The number of nodes in the cache graph')
    parser.add_argument('-d', '--dpg_window', type=int, default=500, help='The number of nodes in the DPG')
    parser.add_argument('-t', '--threshold', type=int, default=config.TEMPLATE_THRESHOLD, help='The threshold of the occurrence number of the path template')

    args = parser.parse_args()
    save = args.save
    save_mode = args.m
    dataset = args.dataset
    subdataset = args.subdataset
    pccache_window = args.pccache_window
    dpg_window = args.dpg_window
    threshold = args.threshold

    # Change path to longer path
    reducer_map[dataset]['save_path'] = os.path.join(reducer_map[dataset]['save_path'], f'pccache_{pccache_window}_dpg_{dpg_window}_threshold_{threshold}')

    print(f'=================CONFIG=================')
    print(f'Dataset: {dataset}')
    print(f'Subdataset: {subdataset}')
    print(f'Save: {save}')
    print(f'Save mode: {save_mode}')
    print(f'CACHE_NODE_WINDOW: {pccache_window}')
    print(f'DPG_NODE_WINDOW: {dpg_window}')
    print(f'TEMPLATE_THRESHOLD: {threshold}')
    print(f'Reducer_map: {reducer_map[dataset]}')
    print(f'========================================')

    start_reducting(dataset, subdataset, save, save_mode)
