from .refine import *
from .queries import Query
from .wordnet import *
from .graph import *
import numpy as np

def createMSQ(list_of_terms, connect_to_wordnet = False):
    if connect_to_wordnet:
        list_of_terms = connect_list_to_wordnet(list_of_terms)
    else:
        list_of_terms = [set([term]) for term in list_of_terms]
    return Query(np.array(list_of_terms))

class conceptDataset:

    def __init__(self, dataset = None, labels = None, connect_to_wordnet = False):
        """
        Initialize the xDataset instance
        dataset: a list of list of objects e.g. [[obj1, obj2], [obj3, obj4]]
        labels: a list of labels for each instance e.g. ["label1", "label2"] if None then the dataset is unlabeled
        connect_to_wordnet: if True then the objects are connected to wordnet
        """
        self.raw_dataset = dataset
        self.raw_labels = labels

        self.connect_to_wordnet = connect_to_wordnet
        self.dataset = dataset
        self.labels = labels

        if self.dataset:
            self.parse_dataset()

    def parse_dataset(self):
        """
        Parse the dataset (from the argument dataset) to create the MSQs
        """

        if len(self.dataset) != len(self.labels):
            raise ValueError("The number of labels must be equal to the number of instances")
        
        self.dataset = {idx: createMSQ(objects, self.connect_to_wordnet) for idx, objects in enumerate(self.dataset)}
        self.labels = {idx: label for idx, label in enumerate(self.labels)}

    def from_txt(self, txt_filename, with_labels, sep = ","):
        """
        Initialize the xDataset instance from a .txt file
        txt_filename: the path to the txt file
        with_labels: if the .txt file contains labels as the last element of each line
        """
        self.dataset = {}
        self.labels = {}
        with open(txt_filename, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                objects = line.split(sep)
                if with_labels:
                    self.labels[idx] = objects[-1]
                    objects = objects[:-1]
                self.dataset[idx] = createMSQ(objects, self.connect_to_wordnet)
        return self


    def retrieve(self, query, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Retrieve: rank the instances based on their distance from the given query
        query: the query to search (Query or just a list of lists of objects)
        """
        
        if not isinstance(query, Query):
            query = createMSQ(query, self.connect_to_wordnet)
        
        distances = {}
        for idx, cand_q in self.dataset.items():
            cost = refine(query, cand_q, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost, verbose= False, return_edits = False)        
            distances[idx] = cost
        
        # sort distances based on the cost (minimum first)
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        return distances


    def explain(self, query, label, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Search for the closest query in the dataset
        query: the query to search (Query or just a list of objects)
        """

        if self.labels is None:
            raise ValueError("The dataset is not labeled")
        
        if not isinstance(query, Query):
            query = createMSQ(query, self.connect_to_wordnet)

        distances = self.retrieve(query, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)

        for idx, cost in distances.items():
            if self.labels[idx] != label:
                return idx, cost

    def find_edits(self, q1, q2, obj_distance = None, addition_cost = None, removal_cost = None, verbose = False):
        """
        Find the edits between 2 queries
        q1: the first query (Query or just a list of objects)
        q2: the second query (Query or just a list of objects)
        """
        if not isinstance(q1, Query):
            q1 = createMSQ(q1, self.connect_to_wordnet)

        if not isinstance(q2, Query):
            q2 = createMSQ(q2, self.connect_to_wordnet)

        return refine (q1, q2, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost, verbose = verbose, return_edits = True)



    def global_explanation(self, queries, labels, obj_distance = None, addition_cost = None, removal_cost = None):
        
        if self.labels is None:
            raise ValueError("The dataset is not labeled")
        
        # we are going to calculate the global explanation
        # for each label we are going to find the closest instance
        # and then we are going to find the edits between the query and the closest instance
        if len(set(labels)) > 1: 
            print ("Warning: the dataset contains more than 1 label!")

        global_explanation = {}

        for query, label in zip(queries, labels):
            resp = self.explain(query, label, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)
            if resp is None:
                continue 
            
            _, edits = self.find_edits(query, self.dataset[resp[0]], obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)
        
            for obj in edits["additions"]:
                for concept in obj:
                    if concept not in global_explanation:
                        global_explanation[concept] = 0
                    global_explanation[concept] += 1

            for obj in edits["removals"]:
                for concept in obj:
                    if concept not in global_explanation:
                        global_explanation[concept] = 0
                    global_explanation[concept] -= 1

            for obj1, obj2 in edits["transf"]:
                for concept in obj1 - obj2:
                    if concept not in global_explanation:
                        global_explanation[concept] = 0
                    global_explanation[concept] -= 1
                
                for concept in obj2 - obj1:
                    if concept not in global_explanation:
                        global_explanation[concept] = 0
                    global_explanation[concept] += 1

        # sort based on the abs value of the explanation
        global_explanation = {k: v for k, v in sorted(global_explanation.items(), key=lambda item: abs(item[1]), reverse = True)}
        return global_explanation


    def __repr__(self):
        return f"xDataset with {len(self.dataset)} instances"

    def __len__(self):
        return len(self.dataset)   
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)
    

class GraphDataset:

    def __init__(self, dataset, labels = None, connect_to_wordnet = False):
        """
        Initialize the xDataset instance
        dataset: a list of list of objects e.g. [[obj1, obj2, rel^obj2], [obj3, obj4]]
        labels: a list of labels for each instance e.g. ["label1", "label2"] if None then the dataset is unlabeled
        connect_to_wordnet: if True then the objects are connected to wordnet
        """

        self.raw_dataset = dataset
        self.raw_labels = labels
        self.connect_to_wordnet = connect_to_wordnet


        self.dataset = dataset
        self.labels = labels

        if self.dataset:
            self.parse_dataset()


    def parse_dataset(self):
        """
        Parse the dataset (from the argument dataset) to create the Graphs
        """

        if len(self.dataset) != len(self.labels):
            raise ValueError("The number of labels must be equal to the number of instances")
        
        self.dataset = {idx: Graph (objects, connect_to_wordnet = self.connect_to_wordnet) for idx, objects in enumerate(self.dataset)}
        self.labels = {idx: label for idx, label in enumerate(self.labels)}

    
    def retrieve(self, query, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Retrieve: rank the instances based on their distance from the given query
        query: the query to search (Graph or just a list of lists of objects)
        """
        if not isinstance(query, Graph):
            query = Graph(query, self.connect_to_wordnet)
        
        distances = {}
        for idx, cand_q in self.dataset.items():
            cost = query.cost(cand_q, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)       
            distances[idx] = cost
        
        # sort distances based on the cost (minimum first)
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        return distances


    def explain(self, query, label, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Search for the closest query in the dataset
        query: the query to search (Graph or just a list of objects)
        """

        if self.labels is None:
            raise ValueError("The dataset is not labeled")
        
        if not isinstance(query, Graph):
            query = Graph(query, self.connect_to_wordnet)

        distances = self.retrieve(query, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)

        for idx, cost in distances.items():
            if self.labels[idx] != label:
                return idx, cost

    def __repr__(self):
        return f"xDataset with {len(self.dataset)} instances"

    def __len__(self):
        return len(self.dataset)   
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)



class xDataset:

    def __init__(self, dataset = None, labels = None, connect_to_wordnet = False, is_graph = False):
        """
        Initialize the xDataset instance
        dataset: a list of list of objects e.g. [[obj1, obj2, rel^obj2], [obj3, obj4]]
        labels: a list of labels for each instance e.g. ["label1", "label2"] if None then the dataset is unlabeled
        connect_to_wordnet: if True then the objects are connected to wordnet
        is_graph: if True then the dataset is a graph dataset otherwise it is a concept dataset
        """
        self.is_graph = is_graph
        self.labels = labels

        if is_graph:
            self.dataset = GraphDataset(dataset, labels, connect_to_wordnet)
        else:
            self.dataset = conceptDataset(dataset, labels, connect_to_wordnet)

    def retrieve(self, query, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Retrieve: rank the instances based on their distance from the given query
        query: the query to search (Query/Graph or just a list of lists of objects)
        """
        return self.dataset.retrieve(query, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)
    
    def explain(self, query, label, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Search for the closest query in the dataset
        query: the query to search (Query/Graph or just a list of objects)
        """
        return self.dataset.explain(query, label, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)
    
    def global_explanation(self, queries, labels, obj_distance = None, addition_cost = None, removal_cost = None):
        """
        Search for the closest query in the dataset. Works only for conceptDataset not for GraphDataset (graphs are not supported yet)
        query: the query to search (Query or just a list of objects)
        """
        if self.is_graph:
            raise (Exception("Global explanation is not supported for GraphDataset but only for conceptDataset!"))
        return self.dataset.global_explanation(queries, labels, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost)
    
    def find_edits(self, q1, q2, obj_distance = None, addition_cost = None, removal_cost = None, verbose = False):
        """
        Find the edits between 2 queries
        q1: the first query (Query or just a list of objects)
        q2: the second query (Query or just a list of objects)
        """
        if self.is_graph:
            raise (Exception("Global explanation is not supported for GraphDataset but only for conceptDataset!"))
        return self.dataset.find_edits(q1, q2, obj_distance = obj_distance, addition_cost = addition_cost, removal_cost = removal_cost, verbose = verbose)


    def __repr__(self):
        return self.dataset.__repr__()
    
    def __len__(self):  
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __iter__(self):
        return iter(self.dataset)
    
    
