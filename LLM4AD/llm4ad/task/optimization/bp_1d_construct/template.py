"""Prompt metadata for the constructive one-dimensional bin-packing task."""

task_description = '''
Given a set of bins and items, iteratively assign one item to feasible bins.
Design a constructive heuristic used in each iteration, with the objective of minimizing the used bins.
'''

method_signature = 'remaining_items, remaining_capacities'
class_args = ''

method_args = '''
determine_next_assignment:
    Args:
        remaining_items: List of the weights of all unpacked items.
        remaining_capacities: List of the remaining capacity of every bin.
    Returns:
        selected_item, selected_bin: The weight of one item in remaining_items
            and the integer index of a bin that can contain it. Return a tuple
            whose bin is None only when no feasible assignment exists.
'''

func_template = '''thought:{...}
```python
Code:
def determine_next_assignment(remaining_items, remaining_capacities):
    ...
```
'''
