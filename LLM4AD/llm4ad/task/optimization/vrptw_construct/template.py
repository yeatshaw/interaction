task_description = "The task involves finding optimal routes for a fleet of vehicles to serve a set of customers, respecting time windows and vehicle capacity constraints. Help me design an algorithm to select the next node in each step."

method_signature = 'current_node, depot, unvisited_nodes, rest_capacity, current_time, demands, distance_matrix, time_windows'
class_args = ''
method_args = '''
select_next_node:
    Args:
        current_node: int, current node ID.
        depot: int, depot ID.
        unvisited_nodes: np.ndarray, feasible unvisited customer IDs.
        rest_capacity: float, remaining vehicle capacity.
        current_time: float, current route time.
        demands: np.ndarray, customer demands.
        distance_matrix: np.ndarray, pairwise travel distances.
        time_windows: np.ndarray, [earliest, latest] service times.
    Returns:
        next_node: int, next customer ID or depot ID.
'''
func_template = '''thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...
    ...
```
'''
