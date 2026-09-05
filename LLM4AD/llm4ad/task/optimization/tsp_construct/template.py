task_description = (
    "Construct a Traveling Salesman Problem tour by selecting the next city "
    "at each step. Visit every city exactly once, return to the starting city, "
    "and minimize total route distance."
)

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
