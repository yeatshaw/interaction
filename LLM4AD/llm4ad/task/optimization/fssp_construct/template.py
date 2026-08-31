task_description = ('Construct one common job permutation for a flow-shop schedule, '
                    'minimizing the completion time on the final machine.')

method_signature = 'scheduled_jobs, unscheduled_jobs, machine_completion_times, processing_times'
class_args = ''
method_args = '''
select_next_job:
    Args:
        scheduled_jobs: np.ndarray, job IDs already placed in permutation order.
        unscheduled_jobs: np.ndarray, candidate job IDs not yet scheduled.
        machine_completion_times: np.ndarray, completion time of the current partial permutation on each machine.
        processing_times: np.ndarray, shape [machine_count, job_count].
    Returns:
        next_job: int, one job ID from unscheduled_jobs.
'''
func_template = '''thought:{...}
```python
Code:
def <method_name>(<method_args>):
    import ...
    ...
```
'''
