'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
'''
Parse json files into triple format
'''

import os
import re
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Two os.path.dirname adds two directory levels
sys.path.append(current_file_path)
import argparse
from utils import timeit
from opr import *
from opr import dataparser
from opr import config
from opr import dataloader
from opr.datasaver import GraphSaver


def extract_path_number(filename: str) -> int:  # Extract the last consecutive number part from filename and convert to integer
    match = re.search(r'\d+$', filename)
    if match:
        return int(match.group())
    else:
        return 0

def get_log_filepaths(dataset_tar_dir_path: str) -> list:
    '''
    Find all files ending with .json or with filename ending in digits from given dataset directory path,
    collect their full paths into a list, then sort and return the list.
    '''
    log_filepaths = []
    for file in os.listdir(dataset_tar_dir_path):
        filesplit = file.split('.')
        if filesplit[-1] == 'json' or filesplit[-1].isdigit():
            log_filepaths.append(os.path.join(dataset_tar_dir_path, file))
    log_filepaths = sorted(log_filepaths, key=extract_path_number)
    return log_filepaths

def get_ground_truth(ground_truth_dir_path: str) -> set:
    '''
    Read data from specified file path and convert to a set
    '''
    with open(ground_truth_dir_path, 'r') as f:
        ground_truth = f.readlines()
    ground_truth_id = set([x.strip() for x in ground_truth])
    return ground_truth_id

@timeit
def parsing_subdataset(dataset: str, subdataset:str):
    global parser_map
    d_parser_map = parser_map[dataset]

    raw_log_path = os.path.join(d_parser_map['raw_log_path'], subdataset)

    save_path = os.path.join(d_parser_map['save_path'], subdataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    db_path = os.path.join(save_path, subdataset + '_entities_index.db')
    with open(db_path, '+a') as f:
        pass
    DParser = d_parser_map['DParser']

    log_filepaths = get_log_filepaths(raw_log_path)
    if config.EXPERIMENT_FLAG:
        # ground_truth_path = d_parser_map['ground_truth_path']
        # ground_truth_id = get_ground_truth(ground_truth_path)
        dparser = DParser(
            mode = 'ca',
            db_name=db_path,
            flag='w',
            experimental_flag=config.EXPERIMENT_FLAG,
            # ground_truth_id=ground_truth_id
        )
    else:
        dparser = DParser(
            mode = 'ca',
            db_name = db_path,
            flag='w',
            experimental_flag=config.EXPERIMENT_FLAG
        )
    total_num = 0
    @timeit
    def parse_single_file(filepath:str):
        nonlocal total_num
        dl = dataloader.DarpaE3DataLoader(filepath)
        parser_file_path = os.path.join(save_path, filepath.split('/')[-1] + '_parser_items.txt')
        gs = GraphSaver(save_mode='oprem', oprem_path=parser_file_path)#save_mode='oprem'
        # gs = GraphSaver(save_mode='neo4j')#save_mode='neo4j'
        i = 0   
        for entry in dl:
            i += 1
            oprem = dparser.parse_single_entry(entry)
            if oprem is not None:
                gs.save_OPREM(oprem)
        gs.close()
        print(f'Num of entries: {i}')
        total_num += i
    
    for filepath in log_filepaths:
        print(f'===Parsing file:{filepath}===')
        parse_single_file(filepath)
    dparser.pop_cache_to_db()

def parsing_dataset(dataset):
    global parser_map
    raw_log_path = parser_map[dataset]['raw_log_path']
    folders = [name for name in os.listdir(raw_log_path) if os.path.isdir(os.path.join(raw_log_path, name))]
    for subdataset in folders:
        print(f'\n======================Parsing subdataset {subdataset}======================')
        parsing_subdataset(dataset, subdataset)

def start_parsing(dataset:str, subdataset:str):
    if 'e3' in dataset:
        if subdataset is None:
            parsing_dataset(dataset)
        else:
            print(f'\n======================Parsing subdataset {subdataset}======================')
            parsing_subdataset(dataset, subdataset)


if __name__ == '__main__':
    parser_map = {
        'e3theia':{
            'raw_log_path': '/home/yanh/Data/theia',
            'ground_truth_path': 'data/groundtruth/threaTrace-groundtruth/theia.txt',
            'save_path': './data/DARPA_Engagement3/preduction_parsed/theia',
            'DParser': dataparser.DarpaE3TheiaParser
        },
        'e3trace':{
            'raw_log_path': '/root/autodl-tmp/Data/trace',
            'ground_truth_path': 'data/groundtruth/threaTrace-groundtruth/trace.txt',
            'save_path': './data/DARPA_Engagement3/preduction_parsed/trace',
            'DParser': dataparser.DarpaE3TraceParser
        },
        'e3cadets':{
            'raw_log_path': '/home/yanh/Data/cadets',
            'save_path': './data/DARPA_Engagement3/preduction_parsed/cadets',
            'DParser': dataparser.DarpaE3CadetsParser
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='e3trace', help='The dataset to parse')
    parser.add_argument('-s', '--subdataset', type=str, default=None, help='The subdataset to parse')
    args = parser.parse_args()
    
    # Start parsing
    start_parsing(args.dataset, args.subdataset)

