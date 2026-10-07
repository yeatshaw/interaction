task_description = (
    "Construct a Traveling Salesman Problem tour by selecting the next city "
    "at each step. Visit every city exactly once, return to the starting city, "
    "and minimize total route distance."
)

template_program = '''
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]

    return next_node
'''

method_signature = "current_node, destination_node, unvisited_nodes, distance_matrix"
class_args = ""

method_args = """
select_next_node:
    Args:
        current_node: int, ID of the current city.
        destination_node: int, ID of the starting and destination city.
        unvisited_nodes: np.ndarray, IDs of all unvisited cities.
        distance_matrix: np.ndarray, pairwise distances between cities.
    Returns:
        next_node: int, one city ID from unvisited_nodes.
"""

func_template = """thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...
    ...
```
"""
