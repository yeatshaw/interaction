task_description = "Now there's a Python class that implements an algorithm to solve the Traveling Salesman Problem (TSP). Given a set of nodes with their coordinates, design a constructive heuristic that visits every node once and returns to the start while minimizing route  length. The algorithm selects one next node at each construction step."

method_signature = (
    'current_node, destination_node, '
    'unvisited_nodes, distance_matrix'
)

class_args = '''
'''

method_args = '''
select_next_node:
    Args:
        current_node: int, ID of the current node.
        destination_node: int, ID of the destination node.
        unvisited_nodes: np.ndarray, Array of IDs of unvisited nodes.
        distance_matrix: np.ndarray, Distance matrix of nodes.
    Returns:
        next_node: int, ID of the next node to visit.
'''

func_template = '''thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...  # Import any necessary libraries here
    ...
```
'''

'''
Task Description: {task_description}
===== parent vs. child =====
...

===== best vs. worst =====
...

The average score of last generation is {avg_score}.
Algorithm {id1} is generated after being guided by {corresponding guide 1}
Algorithm {id1} is generated after being guided by {corresponding guide 2}
'''



#子代代码
'''
Task Description: {task_description}
Here are a few pieces of algorithm code to complete the above tasks.
## Algorithm id1 ##
Code:
```python
...
```
## Algorithm id2 ##
Code:
```python
...
```

Reflect requirement:
1.
2.
'''

#子代代码+子代thought
'''
Task Description: {task_description}
Here are a few pieces of algorithm code and their corresponding thoughts to complete the above tasks.
## Algorithm id1 ##
thought: {...}
Code:
```python
...
```
## Algorithm id2 ##
thought: {...}
Code:
```python
...
```

Reflect requirement:
1.
2.
'''

#子代代码+子代thought+分数
'''
Task Description: {task_description}
Here are a few pieces of algorithm to complete the above tasks.
## Algorithm id1 ##
thought: {...}
score:
Code:
```python
...
```
## Algorithm id2 ##
thought: {...}
score:
Code:
```python
...
```

Reflect requirement:
1.
2.
'''

#父代代码+子代代码
'''
Task Description: {task_description}
Here are the reference algorithm sets from the previous algorithm design and the new algorithms generated from them.
## Evolution path 1 ##
# Reference Algorithm 1 #
Code:
```python
...
```
# Reference Algorithm n #
Code:
```python
...
```
# New Algorithm id #
Code:
```python
...
```
## Evolution path n ##
# Reference Algorithm id #
Code:
```python
...
```
# New Algorithm id #
Code:
```python
...
```

Reflect requirement:
1.
2.
'''

#父代代码+父代thought+子代代码+子代thought
'''
Task Description: {task_description}
Here are the reference algorithm sets from the previous algorithm design and the new algorithms generated from them.
## Evolution path 1 ##
# Reference Algorithm id1 #
thought: {...}
Code:
```python
...
```
# Reference Algorithm id2 #
thought: {...}
Code:
```python
...
```
# New Algorithm id#
thought: {...}
Code:
```python
...
```
## Evolution path n ##
# Reference Algorithm id #
thought: {...}
Code:
```python
...
```
# New Algorithm id #
thought: {...}
Code:
```python
...
```

Reflect requirement:
1.
2.
'''

"""
# Algorithm #
thought: {...}
score: 
Code:
```python
...
```
"""

#best vs. worst
"""
## Best Algorithm ##
thought: {...}
score: 
Code:
```python
...
```
## Worst Algorithm ##
thought: {...}
score: 
Code:
```python
...
```
"""

"""
===== best vs. worst =====
## Best Algorithm ##
...
## Worst Algorithm ##
...
"""

"""
===== reference =====
Here are a few pieces of algorithm to complete the above.
## Algorithm 1 ##
...
## Algorithm n ##
...
"""

"""
===== parent vs. child =====
Here are the reference algorithm sets from the previous algorithm design and the new algorithms generated from them.
## Evolution path 1 ##
# Reference Algorithm 1 #
...
# Reference Algorithm n #
...
# Generated Algorithm #
...
## Evolution path n ##
# Reference Algorithm #
...
# Generated Algorithm #
...
"""


