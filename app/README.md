# Conceptual Edits as Counterfactual Explanations (CECE)

![png](images/logo.png)

**CECE: a powerful Python AI library for generating semantic counterfactual explanations for any machine learning model.**

More information about the framework and the algorithm that the library uses can be found in the paper: [Choose your Data Wisely: A Framework for Semantic Counterfactuals
](https://arxiv.org/pdf/2305.17667.pdf).

## Citation
```bibtex
@article{dervakos2023choose,
  title={Choose your Data Wisely: A Framework for Semantic Counterfactuals},
  author={Dervakos, Edmund and Thomas, Konstantinos and Filandrianos, Giorgos and Stamou, Giorgos},
  journal={arXiv preprint arXiv:2305.17667},
  year={2023}
}
```

# How to use the library
```python
# Install cece
pip install cece
```

**- or -**

```python
git clone https://github.com/geofila/Semantic-Counterfactuals.git
cd CECE
git install .
```


## Use Case 1: Calculate the semantic distance between 2 queries


```python
from cece.queries import *
from cece.refine import *


q1 = Query(np.array([set(["car", "vehicle"]), set(["tree"])]))
q2 = Query(np.array([set(["car"]), set(["tree"]), set(["person"])]))

r = refine (q1, q2,verbose= True, return_edits = True)

print (r)
```

    Remains the same: {'tree'}
    -------------------------------------
    Tranform {'car'} from {'vehicle'} -> set()
    Add: {'person'}
    --------------------------------------
    (2, {'additions': [{'person'}], 'removals': [], 'transf': [({'car', 'vehicle'}, {'car'})]})



```python
from cece.wordnet import *

q1 = Query(np.array([set(connect_term_to_wordnet("car")),
                     set(connect_term_to_wordnet("man")),]))

q2 = Query(np.array([set(connect_term_to_wordnet("woman")),
                     set(connect_term_to_wordnet("truck"),)]))

r = refine (q1, q2,verbose= True, return_edits = True)
print ("Cost: ", r[0])
```

    -------------------------------------
    Tranform {'motor_vehicle.n.01', 'artifact.n.01', 'instrumentality.n.03', 'whole.n.02', 'object.n.01', 'conveyance.n.03', 'wheeled_vehicle.n.01', 'vehicle.n.01', 'self-propelled_vehicle.n.01', 'physical_entity.n.01', 'entity.n.01'} from {'car.n.01'} -> {'truck.n.01'}
    Tranform {'living_thing.n.01', 'adult.n.01', 'whole.n.02', 'organism.n.01', 'person.n.01', 'object.n.01', 'physical_entity.n.01', 'entity.n.01'} from {'man.n.01'} -> {'woman.n.01'}
    --------------------------------------
    Cost:  4



```python

q1 = Query(np.array([connect_term_to_wordnet("tree"),
                     connect_term_to_wordnet("man"),]))

q2 = Query(np.array([connect_term_to_wordnet("man"),
                     connect_term_to_wordnet("truck"),]))

r = refine (q1, q2,verbose= True, return_edits = True)
print ("Cost: ", r[0])
```

    Remains the same: {'man.n.01', 'living_thing.n.01', 'adult.n.01', 'whole.n.02', 'organism.n.01', 'person.n.01', 'object.n.01', 'physical_entity.n.01', 'entity.n.01'}
    -------------------------------------
    Tranform {'entity.n.01', 'whole.n.02', 'physical_entity.n.01', 'object.n.01'} from {'living_thing.n.01', 'organism.n.01', 'vascular_plant.n.01', 'tree.n.01', 'woody_plant.n.01', 'plant.n.02'} -> {'motor_vehicle.n.01', 'instrumentality.n.03', 'truck.n.01', 'conveyance.n.03', 'wheeled_vehicle.n.01', 'vehicle.n.01', 'self-propelled_vehicle.n.01', 'artifact.n.01'}
    --------------------------------------
    Cost:  14


### or even simpler


```python
q1 = Query(np.array(connect_list_to_wordnet(["tree", "man"])))
q2 = Query(np.array(connect_list_to_wordnet(["man", "truck"])))
r = refine (q1, q2,verbose= True, return_edits = True)
print ("Cost: ", r[0])
```

    Remains the same: {'man.n.01', 'living_thing.n.01', 'adult.n.01', 'whole.n.02', 'organism.n.01', 'person.n.01', 'object.n.01', 'physical_entity.n.01', 'entity.n.01'}
    -------------------------------------
    Tranform {'entity.n.01', 'whole.n.02', 'physical_entity.n.01', 'object.n.01'} from {'living_thing.n.01', 'organism.n.01', 'vascular_plant.n.01', 'tree.n.01', 'woody_plant.n.01', 'plant.n.02'} -> {'motor_vehicle.n.01', 'instrumentality.n.03', 'truck.n.01', 'conveyance.n.03', 'wheeled_vehicle.n.01', 'vehicle.n.01', 'self-propelled_vehicle.n.01', 'artifact.n.01'}
    --------------------------------------
    Cost:  14


## Or even simpler


```python
from cece.xDataset import createMSQ

q1 = createMSQ(["tree", "man"], connect_to_wordnet = True)
q2 = createMSQ(["man", "truck"], connect_to_wordnet = True)
r = refine (q1, q2,verbose= True, return_edits = True)
print ("Cost: ", r[0])
```

    Remains the same: {'man.n.01', 'living_thing.n.01', 'adult.n.01', 'whole.n.02', 'organism.n.01', 'person.n.01', 'object.n.01', 'physical_entity.n.01', 'entity.n.01'}
    -------------------------------------
    Tranform {'entity.n.01', 'whole.n.02', 'physical_entity.n.01', 'object.n.01'} from {'living_thing.n.01', 'organism.n.01', 'vascular_plant.n.01', 'tree.n.01', 'woody_plant.n.01', 'plant.n.02'} -> {'motor_vehicle.n.01', 'instrumentality.n.03', 'truck.n.01', 'conveyance.n.03', 'wheeled_vehicle.n.01', 'vehicle.n.01', 'self-propelled_vehicle.n.01', 'artifact.n.01'}
    --------------------------------------
    Cost:  14


## ## Use Case 2: Use it for a Datatset


```python
from cece.xDataset import *


# initialize an instance of the Dataset
ds = xDataset(dataset = [["tree", "man"],
                         ["man", "truck"],
                         ["kitchen", "oven", "refrigerator"], 
                         ["bed", "pillow", "blanket", "woman"],
                         ["sofa", "cushion", "pillow"],],

              labels = ["outdoor", "outdoor", "indoor", "indoor", "indoor"],
              connect_to_wordnet = True)


ds.retrieve(ds.dataset[1])
```




    {1: 0, 0: 14, 4: 28, 3: 29, 2: 36}




```python
for idx, cost in ds.retrieve(ds.dataset[1]).items():
    print (f"Cost: {cost} for '{ds.labels[idx]}' with id: {idx}")
```

    Cost: 0 for 'outdoor' with id: 1
    Cost: 14 for 'outdoor' with id: 0
    Cost: 28 for 'indoor' with id: 4
    Cost: 29 for 'indoor' with id: 3
    Cost: 36 for 'indoor' with id: 2



```python
results = ds.retrieve([ "car", "woman"])

for idx, cost in results.items():
    print (f"Cost: {cost} for '{ds.labels[idx]}' with id: {idx}")
```

    Cost: 4 for 'outdoor' with id: 1
    Cost: 16 for 'outdoor' with id: 0
    Cost: 27 for 'indoor' with id: 3
    Cost: 28 for 'indoor' with id: 4
    Cost: 36 for 'indoor' with id: 2


### Explain Method - Get a Semantic Counterfatual


```python
results = ds.explain([ "car", "woman"], "outdoor")

print (f"Cost: {results[1]} for '{ds.labels[results[0]]}' with id: {results[0]}")
```

    Cost: 27 for 'indoor' with id: 3



```python
from cece.xDataset import *
ds.find_edits(["car", "man"], ["woman", "truck"])
```




    (4,
     {'additions': [],
      'removals': [],
      'transf': [({'artifact.n.01',
         'car.n.01',
         'conveyance.n.03',
         'entity.n.01',
         'instrumentality.n.03',
         'motor_vehicle.n.01',
         'object.n.01',
         'physical_entity.n.01',
         'self-propelled_vehicle.n.01',
         'vehicle.n.01',
         'wheeled_vehicle.n.01',
         'whole.n.02'},
        {'artifact.n.01',
         'conveyance.n.03',
         'entity.n.01',
         'instrumentality.n.03',
         'motor_vehicle.n.01',
         'object.n.01',
         'physical_entity.n.01',
         'self-propelled_vehicle.n.01',
         'truck.n.01',
         'vehicle.n.01',
         'wheeled_vehicle.n.01',
         'whole.n.02'}),
       ({'adult.n.01',
         'entity.n.01',
         'living_thing.n.01',
         'man.n.01',
         'object.n.01',
         'organism.n.01',
         'person.n.01',
         'physical_entity.n.01',
         'whole.n.02'},
        {'adult.n.01',
         'entity.n.01',
         'living_thing.n.01',
         'object.n.01',
         'organism.n.01',
         'person.n.01',
         'physical_entity.n.01',
         'whole.n.02',
         'woman.n.01'})]})



### Global Explanations 


```python
from xDataset import *

# initialize an instance of the Dataset
ds = xDataset(dataset = [["tree", "man"],
                         ["man", "truck"],
                         ["kitchen", "oven", "refrigerator", "man"], 
                         ["bed", "blanket", "woman"],
                         ["sofa", "pillow", "man"],],

              labels = ["outdoor", "outdoor", "indoor", "indoor", "indoor"],
              connect_to_wordnet = True)

ds.global_explanation([["tree", "man", "car"], ["man", "truck", "car"]], ["outdoor", "outdoor"])
```




    {'motor_vehicle.n.01': -3,
     'conveyance.n.03': -3,
     'wheeled_vehicle.n.01': -3,
     'vehicle.n.01': -3,
     'self-propelled_vehicle.n.01': -3,
     'padding.n.01': 2,
     'cushion.n.03': 2,
     'pillow.n.01': 2,
     'car.n.01': -2,
     'sofa.n.01': 2,
     'furnishing.n.02': 2,
     'seat.n.03': 2,
     'furniture.n.01': 2,
     'living_thing.n.01': -1,
     'organism.n.01': -1,
     'vascular_plant.n.01': -1,
     'tree.n.01': -1,
     'woody_plant.n.01': -1,
     'plant.n.02': -1,
     'artifact.n.01': 1,
     'truck.n.01': -1,
     'instrumentality.n.03': -1}




```python
# plot the global explanation in abar plot
import matplotlib.pyplot as plt

explanations = ds.global_explanation([["tree", "man", "car"], ["man", "truck", "car"]], ["outdoor", "outdoor"])

plt.bar(explanations.keys(), explanations.values())
plt.xticks(rotation=90)
plt.show()
```


    
![png](images/global_explanations_example.png)
    



```python

```