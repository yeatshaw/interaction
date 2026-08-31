task_description = ('Construct capacity-feasible CVRP routes by selecting the next customer '
                    'at each step, minimizing total travel distance.')

method_signature = 'current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix'
class_args = ''
method_args = '''
select_next_node:
    Args:
        current_node: int, current node ID.
        depot: int, depot node ID.
        unvisited_nodes: np.ndarray, feasible unvisited customer IDs.
        rest_capacity: float, remaining vehicle capacity.
        demands: np.ndarray, demand of every node.
        distance_matrix: np.ndarray, pairwise travel distances.
    Returns:
        next_node: int, one ID from unvisited_nodes or depot.
'''
func_template = '''thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...
    ...
```
'''
