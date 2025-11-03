'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
from typing import Dict, Tuple, List, Set, Optional
from collections import Counter
from pathlib import PurePath
import json
import os

__all__ = ["PathTrieTree", "NoiseMatchTemplate"]

# @dataclass
# class PathTrieNode:
#     name: str
#     parent: 'PathTrieNode'
#     successors: Dict[str, 'PathTrieNode'] = field(default_factory=dict)
#     occurr_num: int = 0

class PathTrieNode:
    def __init__(self, name: str = None):
        '''Node of the PathTrieTree.
        
        Parameters
        ------
        name: str
            a single path level name of the node.
            e.g. 'bin', 'bash', 'syslog' in '/bin/bash/syslog'
        parent: PathTrieNode
            The parent node of the node.
        '''
        self.name = name
        self.successors = {}  # {name: PathTrieNode, ...}
        self.occurr_num = 1  # the occurrence number of the node
    
    def add_successor(self, name: str, node):
        if name not in self.successors:
            self.successors[name] = node
    
    def has_successor(self, name: str):
        return name in self.successors

class PathTrieTree:
    def __init__(self) -> None:
        '''Hierarchy path trie tree for process related template extraction in CacheGraph'''
        # root node of the trie tree
        self.root: PathTrieNode = PathTrieNode(name='PTT_ROOT')  
        # [{nodename1: PathTrieNode, nodename2: PathTrieNode, ...}, {}, ...]
        # index 0 is the root node, index 1 is the first level node, and so on.
        # NOTE EVERY SAME LEVEL has a group of DIFFERENT nodes.
        # DIFFERENT LEVEL can have the SAME node name.
        self.hierapath_map: List[Dict] = [{'PTT_ROOT': self.root}]
        # all paths in the trie tree
        self.paths = []

    @staticmethod
    def path_to_words(path):
        words = path.split('/')
        if words[0] == '':
            words = words[1:]  # if the path is absolute path
        return words

    @staticmethod
    def count_and_filter(templates: List[str], threshold: int) -> List[str]:
        """
        Count the occurrence of the templates and filter the 
        templates with occurrence less than threshold
        """
        # TODO Template expansion
        # e.g. two template: /bin/bash ---(can merged to)--> /bin/*
        # but n^2 time complexity
        counts = Counter(templates)
        return [element for element, count in counts.items() if count >= threshold]

    def add_node(self, name: str, parent: PathTrieNode, hieranum: int) -> PathTrieNode:
        node = PathTrieNode(name=name)
        parent.add_successor(name, node)
        self.hierapath_map[hieranum][name] = node
        return node

    def add_path(self, path: str):
        '''NOTE the template should start with '/' '''
        self.paths.append(path)
        pwords = self.path_to_words(path)
        
        node: PathTrieNode = self.root
        for i in range(1, len(pwords)+1):  # start from 1
            if i == len(self.hierapath_map):  # add new level
                self.hierapath_map.append({})

            word = pwords[i-1]
            if word not in self.hierapath_map[i]:  # if word not in i level
                node = self.add_node(word, node, i)
            else:  # if word in i level
                if not node.has_successor(word):  # if word not in node successors
                    node.add_successor(word, self.hierapath_map[i][word])
                node = node.successors[word]
                node.occurr_num += 1

    def extract_path_template(self, threshold: int = 5):
        """Extract the path template from the paths in the trie tree.
        
        Parameters
        ------
        threshold: int
            The threshold of the occurrence number of the path template.
        """
        templates = []
        for p in self.paths:
            templates.append(self.extract_single_path_template(p))
        return self.count_and_filter(templates, threshold)

    def extract_single_path_template(self, path: str) -> str:
        """
        Extract the path template from a single path in the trie tree.
        Converting node with minimum occurrence number to '*', without some special cases.
        """        
        words: List[str] = self.path_to_words(path)
        words_occrr_num: List[int] = []
        node: PathTrieNode = self.root
        for word in words:
            if word in node.successors:
                node = node.successors[word]
                words_occrr_num.append(node.occurr_num)
            else:
                raise ValueError(f'Node not in the trie tree: {word}')
        
        mini_in_path_num = min(words_occrr_num)
        template_path_list = words
        t_flag = False
        for i in range(len(template_path_list)):
            # if the node has the minimum occurrence number,
            # and one of the previous node has more than the minimum occurrence number.
            # NOTE this can avoid * from the beginning of the path.
            if words_occrr_num[i] == mini_in_path_num and t_flag:
                template_path_list[i] = '*'
            elif words_occrr_num[i] > mini_in_path_num:
                t_flag = True
        return os.path.join('/', *template_path_list)


class NoiseMatchTemplate(dict):
    def __init__(self, init_templates: Dict[str, Set] = None, template_path: str = None):
        '''Process centric template matching for noise extraction.
        {key1: Set{template1, template2, ...}, ...}

        Parameters
        ------
        templates: Dict
            The init templates. 
        template_path: str
            The path of the template file. If the templates is not empty, this will be ignored.
        '''
        if init_templates is not None:
            self.update(init_templates)
        elif template_path is not None:
            self.load_json(template_path)
    
    def __str__(self) -> str:
        return super().__str__()
    
    def add_templates(self, match_temps: Dict):
        for key, temps in match_temps.items():
            for temp in temps:
                self.add_template(key, temp)
    
    def add_template(self, key: str, template: str):
        if key not in self:
            self[key] = set()
        self[key].add(template)
    
    def delete_template(self, key: str, template: str):
        '''
        Delete the template from the key if the template is in the key.
        Else, do nothing.
        '''
        temps: set = self.get(key, set())
        temps.discard(template)

    def can_match_template(self, key: str, path: str) -> bool:
        '''Check if the path can match template of the key or not.'''
        if key not in self:
            return False
        for template in self[key]:
            if PurePath(path).match(template):
                return True
        return False

    def match_template(self, key: str, path: str) -> Optional[str]:
        if key not in self:
            return None
        for template in self[key]:
            if PurePath(path).match(template):
                return template
        return None

    def load_json(self, filepath: str):
        with open(filepath, 'r') as f:
            load_dict = dict(json.load(f))
        for key, temps in load_dict.items():
            self.add_templates({key: set(temps)})

    def save_json(self, filepath: str):
        res_dict = {}
        for key, temps in self.items():
            res_dict[key] = list(temps)
        with open(filepath, 'w') as f:
            json.dump(res_dict, f, indent=4)
