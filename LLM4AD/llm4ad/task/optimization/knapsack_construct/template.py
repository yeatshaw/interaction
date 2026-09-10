# Prompt metadata consumed by EoH/MCTS-Recipe.  The executable template is
# intentionally omitted, as in tsp_construct; get_info() creates a minimal
# function template from method_signature.
method_signature = 'remaining_capacity, remaining_items'
class_args = ''
method_args = '''
select_next_item:
    Args:
        remaining_capacity: Remaining capacity of the knapsack.
        remaining_items: Candidate tuples (weight, value, original_index).
    Returns:
        selected_item: One tuple from remaining_items, or None if no item fits.
'''
func_template = '''thought:{...}
```python
Code:
def select_next_item(remaining_capacity, remaining_items):
    ...
```
'''

task_description = '''
Given a set of items with weights and values, the goal is to select a subset of items
that maximizes the total value while not exceeding the knapsack's capacity.
Help me design a novel algorithm to select the next item in each step.
'''
