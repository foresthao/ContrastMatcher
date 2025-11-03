'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import networkx as nx
from typing import Dict, List, Union
from opr import config
from opr.logger import logger

__all__ = ["OPREventModel", "OPRNodeModel", "OPREdgeModel"]

class OPREventModel(dict):
    def __init__(self, u: Dict=None, v: Dict=None, e: Dict=None):
        super().__init__()
        if u is None: u={}
        if v is None: v={}
        if e is None: e={}
        self.u = OPRNodeModel(**u)
        self.v = OPRNodeModel(**v)
        self.e = OPREdgeModel(**e)
        self.update({
            'u': self.u,
            'v': self.v,
            'e': self.e
        })

        if u and v:
            self.update_sub_and_obj()
    
    #---------------UPDATE------------------
    def update_from_graph_edge(self, g: Union[nx.Graph, nx.MultiGraph], u,v,k = None):
        # Update from networkx graph
        u_node = g.nodes[u]
        v_node = g.nodes[v]
        if g.is_multigraph():
            edge = g.edges[u, v, k]
        else:
            edge = g.edges[u, v]
        self.update_e(**edge)
        self.update_u(**u_node)
        self.update_v(**v_node)
        self.update_sub_and_obj()

    def update_from_loprem(self, loprem: List):
        # Update from oprem model
        u_vid, u_nid, u_type, u_content, u_flag, \
        v_vid, v_nid, v_type, v_content, v_flag, \
        edge_type, edge_ts, edge_te =loprem
        u_flag = int(u_flag)
        v_flag = int(v_flag)
        edge_ts = int(edge_ts)
        edge_te = int(edge_te)
        self.update_u(vid=u_vid, nid = u_nid, type = u_type, content= u_content, flag = u_flag)
        self.update_v(vid=v_vid, nid = v_nid, type = v_type, content= v_content, flag = v_flag)
        self.update_e(type = edge_type, ts = edge_ts, te = edge_te)
        self.update_sub_and_obj()

    def update_sub_and_obj(self):
        # Update event from source and destination nodes
        if self.u['type'] == config.CENTRAL_NODE_TYPE:
            self.sbj = self.u
            self.obj = self.v
        elif self.v['type'] == config.CENTRAL_NODE_TYPE:
            self.sbj = self.v
            self.obj = self.u
        else:
            logger.info(f"Missing central nodes: {self}")

    def update_sbj(self, **kwargs):
        self.sbj.update_node(**kwargs)

    def update_obj(self, **kwargs):
        self.obj.update_node(**kwargs)

    def update_u(self, **kwargs):
        self.u.update_node(**kwargs)

    def update_v(self, **kwargs):
        self.v.update_node(**kwargs)

    def update_e(self, **kwargs):
        self.e.update_edge(**kwargs)

    # Query section
    # ============QUERY=========== #
    def get_sbj(self, key: str):
        return self.sbj.get(key)
    
    def get_obj(self, key: str):
        return self.obj.get(key)

    def get_u(self, key: str):
        return self['u'].get(key)
    
    def get_v(self, key: str):
        return self['v'].get(key)
    
    def get_e(self, key: str):
        return self['e'].get(key)
    
    def get_(self, key1: str, key2: str):
        try:
            return self[key1][key2]
        except KeyError:
            return None

    
class OPRNodeModel(dict):
    def __init__(self, **kwargs):
        super().__init__()
        init_dict = {
            'vid' : None,
            'nid' : None,
            'type': None,
            'content': None,
            'flag': None
        }
        self.update(init_dict)
        self.update_node(**kwargs)
    
    def update_node(self, **kwargs):
        for k, v in kwargs.items():
            if k in self:
                self[k] = v


class OPREdgeModel(dict):
    def __init__(self, **kwargs):
        super().__init__()

        init_dict = {
            'type': None, #str
            'ts': None, #int
            'te': None #int
        }
        self.update(init_dict)
        self.update_edge(**kwargs)
    
    def update_edge(self, **kwargs):
        for k, v in kwargs.items():
            if k in self:
                self[k] = v