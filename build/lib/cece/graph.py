from .refine import *
from .queries import *
from .wordnet import *

import functools


def node_transition_cost(obj1, obj2, obj_distance = None, addition_cost = None, removal_cost = None):
    return refine(obj1, obj2, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)

def add_cost(concept, obj_distance = None, addition_cost = None, removal_cost = None):
    return refine(concept, Query(np.array([])), obj_distance= obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)

def remove_cost(concept, obj_distance = None, addition_cost = None, removal_cost = None):
    return refine(Query(np.array([])), concept, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)


class Graph:
    def __init__(self, objs, connect_to_wordnet = False):
        self.objs = objs # objects should be a list of lists
        if connect_to_wordnet:
            self.concepts = [Query(np.array(connect_list_to_wordnet(obj))) for obj in self.objs]
        else:
            self.concepts = [Query(np.array(obj)) for obj in self.objs]
    
    def cost(self, g2, obj_distance = None, addition_cost = None, removal_cost = None, return_edits = False):
        return refine (self, g2, 
                       obj_distance = functools.partial(node_transition_cost, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost),
                       addition_cost = functools.partial(add_cost, obj_distance = obj_distance, addition_cost= addition_cost, removal_cost= removal_cost), 
                       removal_cost = functools.partial(remove_cost, obj_distance = obj_distance, addition_cost= addition_cost, removal_cost= removal_cost), 
                       return_edits = return_edits)

    def __repr__(self):
        return f"Graph: {self.concepts}"

    def __str__(self):
        return f"Graph: {self.concepts}"

    def __getitem__(self, idx):
        return self.concepts[idx]
    