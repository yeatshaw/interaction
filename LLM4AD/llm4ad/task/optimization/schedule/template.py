task_description = '''
Construct an initial placement on a circular grid of N slots. The input is a
list of distinct positive service IDs and their required counts. Return a list
P of exactly N entries: each service ID x must occur s_x times, and 0 must
occur N minus the sum of service counts times.

For each represented type x, including 0 when present, let g_i be the circular
distance between consecutive occurrences and define
D_x = sum_i abs(g_i - N/s_x). Its worst-case deviation is
D_x_worst = 2*(s_x-1)*(N/s_x-1). Set u_x = 1 when s_x <= 1 or s_x == N;
otherwise set u_x = 1 - D_x/D_x_worst. The layout uniformity is
U(P) = sum_x (s_x/N)*u_x.

This initialization-only search uses fitness 0.55*U(P), matching the
uniformity weight in the full problem. Initialization has no previous grid,
so stability is not scored here; the Python runtime is not the C++ speed score.

The generated algorithm must use only integer state and addition, subtraction,
comparisons, bitwise operations, and shifts. Multiplication, division,
remainder, floating point, and math library functions are forbidden in the
algorithm. The evaluator computes the mathematical score separately.
'''

template_program = '''
def init(N, service_list):
    """Build and return a length-N grid matching the requested service counts.

    Args:
        N: Number of slots in the circular grid.
        service_list: Distinct (service_id, count) pairs; service IDs are positive.

    Returns:
        A list of N entries. Each service appears exactly count times; all
        remaining entries are 0 (vacant).
    """
    grid = []
    for _ in range(N):
        grid.append(0)

    position = 0
    for service_id, count in service_list:
        remaining = count
        while remaining > 0:
            grid[position] = service_id
            position += 1
            remaining -= 1
    return grid
'''

method_signature = 'N, service_list'
class_args = ''

method_args = '''
init:
    Args:
        N: int, 1 <= N <= 10000.
        service_list: Up to 100 distinct (service_id, count) pairs. IDs and
            counts are positive integers; the sum of counts is at most N.
    Returns:
        A list of exactly N integers. Each input service ID occurs exactly
        count times, and all remaining entries are 0.
'''

func_template = '''thought:{...}
```python
Code:
def init(N, service_list):
    ...
```
'''
