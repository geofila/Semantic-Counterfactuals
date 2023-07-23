import networkx as nx
from networkx.algorithms import bipartite



def default_obj_distance (obj1, obj2):
    lca = obj1.intersection(obj2)
    diffs = len(obj1 - lca) + len(obj2 - lca)
    return diffs

def default_addition_cost (obj):
    return len(obj)

def default_removal_cost (obj):
    return len(obj)



def refine (q1, q2, obj_distance = None, addition_cost = None, removal_cost = None, verbose= False, return_edits = False):
    # q1 = msq[id1] # get msq (most specific query) for each id
    # q2 = msq[id2] # get msq for each id

    # if one of the distances is not given we use the default
    if obj_distance is None:
        obj_distance = default_obj_distance
    if addition_cost is None:
        addition_cost = default_addition_cost
    if removal_cost is None:
        removal_cost = default_removal_cost

    # we are going to calculate the cost of refine between 2 queries
    edits = {"additions": [], "removals": [], "transf": []}
    objects1 = {i: c for i, c in enumerate (q1.concepts)} # give to each object of q1 a unique id 
    objects2 = {i + len(objects1): c for i, c in enumerate (q2.concepts)} # give to each object of q2 a unique id

    # which concepts are common between the 2 instances
    matches = {}
    for i, c1 in objects1.items():
        for j, c2 in objects2.items():
            if obj_distance(c1, c2) == 0 and j not in matches.values(): # if the objects are the same and this id is not already in the match, ie if the object has not already been matched with other object
                matches[i] = j
                break # stop searching because maybe the same object exists 2 times in the instance e.g. 2 persons or 2 times the same word

    # the match items must be removed from the list of items
    for i, j in matches.items():
        if verbose:
            print (f"Remains the same: {objects1[i]}")
        objects1.pop(i, None) # remove this object from the list of items of q1
        objects2.pop(j, None) # remove this object from the list of items of w2

    if verbose:
        print ("-------------------------------------")

    cost = 0 # we initialize the cost of transitions
    # now we are going to calculate the matches of the objects
    if len(objects1) != 0 and len(objects2) != 0: # if one of the 2 is empty we do not have to match so we just go and remove or add the objects of the other image
        B = nx.Graph() # create two bipartite graphs
        B.add_nodes_from(objects1.keys(), bipartite=0) # nodes are the objects in instance 1
        B.add_nodes_from(objects2.keys(), bipartite=1) # nodes are the objects in instance 2
        # we want to add edges from obj in instance 1 with obj in instance 2
        # if a transition is not valid e.g. transform a old man to a young one
        # we just do not connect these 2 nodes

        transition_matrix = {}
        for i, obj1 in objects1.items(): # for each object in instance 1
            for j, obj2 in objects2.items(): # for each object in instance 2
                weight = obj_distance(obj1, obj2) # calculate the cost of the transition
                if weight != 10e6:
                    transition_matrix[f"{i}-{j}"] = weight
                    B.add_edge(i, j, weight = weight) # add edge

        
        if len(transition_matrix) != 0:
            matches = bipartite.matching.minimum_weight_full_matching(B, objects1.keys(), "weight") # calculate minimum weight full matching
            for i, j in matches.items():
                if i in objects1: # to do it once because the matches return matches from both image 1 -> 2 and from 2 -> 1
                    if transition_matrix[f"{i}-{j}"] != 10e6: # if the transition is valid
                        cost += transition_matrix[f"{i}-{j}"] # add the cost for this transition
                        edits["transf"].append((objects1[i], objects2[j]))
                        if verbose: # print the tranformation of the objects
                            n1 = objects1[i]
                            n2 = objects2[j]
                            print (f"Tranform {n1.intersection(n2)} from {n1 - n2} -> {n2 - n1}")
                        # and remove the item from the list of items in each image
                        objects1.pop(i, None)
                        objects2.pop(j, None)

    for i, obj in objects1.items(): # any items remaining in instance 1 must be removed
        cost += removal_cost(obj) #len(obj)
        edits["removals"].append(obj)
        if verbose:
            print (f"Remove: {obj}")

    for i, obj in objects2.items(): # any items remaining in instance 1 must be added
        cost += addition_cost(obj) #len(obj)
        edits["additions"].append(obj)
        if verbose:
            print (f"Add: {obj}")

    if verbose:
        print ("--------------------------------------")
        
    if return_edits:
        return cost, edits
    return cost
