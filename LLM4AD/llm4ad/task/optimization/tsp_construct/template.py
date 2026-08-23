task_description = (
    "Given a set of nodes with their coordinates, design a constructive heuristic "
    "that visits every node once and returns to the start while minimizing route "
    "length. The algorithm selects one next node at each construction step."
)

method_signature = (
    'current_node: int, destination_node: int, '
    'unvisited_nodes: np.ndarray, distance_matrix: np.ndarray'
)

class_args = '''
- current_node: int, ID of the node currently being visited.
- destination_node: int, ID of the destination/start node.
- unvisited_nodes: np.ndarray, IDs of nodes that have not been visited.
- distance_matrix: np.ndarray, Pairwise distances between all nodes.
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
import numpy as np
def <method_name>(<method_args>):
    ...
```
'''
