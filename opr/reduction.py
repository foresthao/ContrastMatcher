'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
from typing import List, Optional

from opr import config, utils
from opr import OPREventModel, NoiseMatchTemplate, OPRNodeModel
from opr import DPGraph, FDPGraph, CacheGraph
from opr.logger import logger

class OnlineReduction:
    def __init__(self, info_preserve_method: str='cp'):
        '''
        Build path dependency graph DPG
        '''
        self.info_preserve_method = info_preserve_method
        if info_preserve_method == 'cp':
            self.dpg = DPGraph(
                node_window = config.DPG_NODE_WINDOW,
                central_node_type = config.CENTRAL_NODE_TYPE
            )
        elif info_preserve_method == 'fdp':
            self.dpg = FDPGraph(
                node_window = config.DPG_NODE_WINDOW,
                central_node_type = config.CENTRAL_NODE_TYPE
            )
        self.pccache = CacheGraph(
                node_window = config.DPG_NODE_WINDOW,
                central_node_type = config.CENTRAL_NODE_TYPE
        )
        
        self.templates = NoiseMatchTemplate(config.INIT_TEMPLATES, config.INIT_TEMPLATE_PATH)

    def consume_single_OPREM(self, oprem:OPREventModel) -> Optional[List[List[OPREventModel]]]:
        '''
        Add a single OPREM to DPG or PCCache. Subject node is the central node, so only need to check object node.
        '''
        obj = oprem.obj
        obj_type = obj['type']
        obj_nid = obj['nid']

        if obj_type == 'NetFlowObject':
            self.dpg.net_redcution(oprem)
        elif obj_type == config.CENTRAL_NODE_TYPE:  
            self.dpg.central_node_reduction(oprem)
        elif obj_type == 'FileObject':
            self.pccache.add_from_OPREM(oprem)
            # EXTRACTION! if the object node has more than one neighbors
            # (include successors and predecessors) in pccache, extract 
            # the component to DPG
            negb = utils.unique_neighbors(self.pccache, obj_nid)  # Not implemented
            if len(negb) > 1:
                logger.info(f'---E1 MULTIPLE LINKS---')  # Handle multiple links case
                all_poped_oprem = self.pop_pccache_multi_node_and_update_template(obj_nid, negb)
                self.update_dpg_from_ft_oprems(all_poped_oprem)
            # EXTRACTION! if the node size of pccache reach the node_window,
            # extract the latest recent subgraph to DPG
            while self.pccache.number_of_nodes() > self.pccache.node_window:
                logger.info(f'---E2 NODE WINDOW---')  # Handle window size exceeded case
                pccahe_oldest_central_node = self.pccache.central_nodes.get()
                # NOTE if pccache has node in the central node
                # because some central nodes may be extracted to DPG before
                # like, E1 MULTIPLE LINKS
                if self.pccache.has_node(pccahe_oldest_central_node):
                    all_poped_oprem = self.pop_pccache_central_node_and_update_template(pccahe_oldest_central_node)
                    self.update_dpg_from_ft_oprems(all_poped_oprem)

        # If the node size of DPG reach the node_window, extract the oldest node. If DPG graph exceeds window size, extract oldest node
        all_node_events = []
        while self.dpg.number_of_nodes() > self.dpg.node_window:
            oldest_vid = self.dpg._vid_queue.get()
            oldest_type = self.dpg.nodes[oldest_vid]['type']
            oldest_nid = self.dpg.nodes[oldest_vid]['nid']
            # EXTRACTION! if the node is the latest version of a central node, 
            # and its nid in pccache. extract the component from pccache to DPG
            if oldest_type==config.CENTRAL_NODE_TYPE and self.dpg.is_latest_vid(oldest_vid) and self.pccache.has_node(oldest_nid):
                logger.info(f'---E3 CENTRAL NODE POP---')  # Central node pop
                all_poped_oprem = self.pop_pccache_central_node_and_update_template(oldest_nid)
                self.update_dpg_from_ft_oprems(all_poped_oprem)
            # pop the oldest node from the DPG
            all_node_events.append(self.dpg._pop_node(oldest_vid))
        return all_node_events

    def remove_degree_zero_nodes(self):
        """Remove nodes with degree 0 from the DPG."""
        nodes_to_remove = [node for node, degree in self.dpg.degree() if degree == 0]
        for node in nodes_to_remove:
            self.dpg.remove_node(node)

    def remove_nodes_with_small_degree(self, threshold):
        """Remove nodes with degree less than the specified threshold."""
        nodes_to_remove = [node for node, degree in self.dpg.degree() if degree < threshold]
        for node in nodes_to_remove:
            self.dpg.remove_node(node)
    
    def update_dpg_from_ft_oprems(self, oprems: List[OPREventModel]) -> None:
        '''
        Update DPG from file nodes and templates
        '''
        for _opr in oprems:
            if _opr.obj['type'] == 'FileObject':
                self.dpg.file_reduction(_opr)
            elif _opr.obj['type'] == 'Template':
                self.dpg.template_reduction(_opr)

    def pop_pccache_to_dpg(self) ->None:
        '''Pop all nodes from DPG graph'''
        pass


    def pop_pccache_central_node_and_update_template(self, central_node_nid: str) -> List[OPREventModel]:
        '''Pop central node and its related edges from cache, convert connected components to templates, return event list of this node'''
        subg = self.pccache.pop_weakly_connected_components(central_node_nid)
        # match templates for all F, if matched, change F to T
        subg_without_t = subg.copy()
        central_node_content = subg.nodes[central_node_nid]['content']
        for nid, n_data in subg.nodes(data=True):
            if nid != central_node_nid:
                mt = self.templates.match_template(central_node_content, n_data['content'])
                if mt:
                    # TODO merge Template nodes here
                    # No need to allocate nid, because nid will allocate in template reduction
                    subg.nodes[nid]['content'] = mt
                    subg.nodes[nid]['type'] = 'Template'
                    subg_without_t.remove_node(nid)
        # extract template from F
        temps = utils.graph_to_template(subg_without_t)
        # if temps is not empty, update the templates and the properties of the nodes
        if temps:
            self.templates.add_templates(temps)
            for nid, n_data in subg_without_t.nodes(data=True):
                if nid != central_node_nid:
                    mt = self.templates.match_template(central_node_content, n_data['content'])
                    if mt:
                        subg.nodes[nid]['content'] = mt
                        subg.nodes[nid]['type'] = 'Template'
        del subg_without_t

        # merge multi edges for T and F
        merged_subg = utils.merge_multi_edges(subg)
        del subg
        # convert to OPREventModel and sort
        all_poped_oprem = []
        for u, v, e_data in merged_subg.edges(data=True):
            oprem = OPREventModel()
            oprem.update_from_graph_edge(merged_subg, u, v, e_data['type'])
            all_poped_oprem.append(oprem)
        all_poped_oprem = utils.sort_oprems(all_poped_oprem)
        return all_poped_oprem

    def pop_pccache_multi_node_and_update_template(self, multi_node_nid: str, neighbors: set) -> List[OPREventModel]:
        '''Pop multi-link node and its related edges from cache, convert connected components to templates'''
        # pop the multi link node and its related edges from the Cache
        all_poped_oprem = utils.single_hop_to_oprem(self.pccache, multi_node_nid)
        self.pccache.remove_node(multi_node_nid)
        # pop the central node and its related edges from the Cache
        for central_node_nid in neighbors:
            all_poped_oprem += self.pop_pccache_central_node_and_update_template(central_node_nid)
        all_poped_oprem = utils.sort_oprems(all_poped_oprem)
        return all_poped_oprem 