task_description = "Now there's a Python class that implements an algorithm to solve the Traveling Salesman Problem (TSP) based on constructive heuristics."

method_signature = (
    'current_node: int, destination_node: int, '
    'unvisited_nodes: np.ndarray, distance_matrix: np.ndarray'
)

class_args = '''
'''

method_args = '''
select_next_node:
    Args:
        current_node: int, ID of the current node.
        destination_node: int, ID of the destination/start node.
        unvisited_nodes: np.ndarray, IDs of all unvisited nodes.
        distance_matrix: np.ndarray, Pairwise node distance matrix.
    Returns:
        next_node: int, The ID of exactly one node from unvisited_nodes.
'''

func_template = '''thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...  # Import any necessary libraries here
    ...
```
'''

#子代代码
'''
Task Description: {task_description}
Here are a few pieces of algorithm code to complete the above tasks.
## Algorithm 1 ##
Code:
```python
...
```
## Algorithm n ##
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
## Algorithm 1 ##
thought: {...}
Code:
```python
...
```
## Algorithm n ##
thought: {...}
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
# Reference Algorithm #
Code 1:
```python
...
```
Code n:
```python
...
```
# New Algorithm #
Code:
```python
...
```
## Evolution path n ##
# Reference Algorithm #
Code:
```python
...
```
# New Algorithm #
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
# Reference Algorithm #
thought 1: {...}
Code 1:
```python
...
```
thought n: {...}
Code n:
```python
...
```
# New Algorithm #
thought: {...}
Code:
```python
...
```
## Evolution path n ##
# Reference Algorithm #
thought: {...}
Code:
```python
...
```
# New Algorithm #
thought: {...}
Code:
```python
...
```

Reflect requirement:
1.
2.
'''
