task_description = "Now there's a Python class that implements an algorithm based on the L-SHADE algorithm without LPSR and history record of CR to solve a multimodal, ill-conditioned 30-dimensional black-box."

class_args = '''
- n_individuals: int, Number of individuals in the population.
- ndim_problem: int, Dimension of the problem.
- h: int, Length of historical memory.
- p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
- m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
- lower_boundary: float, Lower boundary of the problem.
- upper_boundary: float, Upper boundary of the problem.
- max_function_evaluations: int, Maximum number of function evaluations.
- initial_pop_size: int, Initial population size.
- _n_generations: int, Current number of generations.
- stagnation_threshold: int, The number of consecutive generations without improvement considered as stagnation
- stagnation_tol: float, relative improvement threshold
- _stagnation_counter: int, Count how many consecutive generations the fitness has not improved
- _evaluate_fitness: method, can be used to calculate the fitness, the shape of input is (1, self.ndim_problem), the output is a float
- rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
'''

method_args = '''
mutate:
    Args:      
        x: np.array, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.array, The fitness of current population of individuals, shape=(self.n_individuals,).
        a: np.array, External archive used in lshade, shape=(n, self.ndim_problem).
    Returns:
        x_mu: np.array, Population individuals after mutation, shape=(self.n_individuals, self.ndim_problem).
        f_mu: np.array, Scaling factor F used during mutation, shape=(self.n_individuals,).
        r: np.array, Index for selecting the scaling factor F and crossover rate CR, shape=(self.n_individuals,)

crossover:
    Args:
        x_mu: np.array, Donor population after mutation, shape=(self.n_individuals, self.ndim_problem).
        x: np.array, Parent population, shape=(self.n_individuals, self.ndim_problem).
        r: np.array, Parameter-memory indices from mutation, shape=(self.n_individuals,).
    Returns:
        x_cr: np.array, Trial population after crossover, shape=(self.n_individuals, self.ndim_problem).
        p_cr: np.array, Crossover probabilities, shape=(self.n_individuals,).

restart:
    Args:
        x: np.array, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.array, The fitness of current population of individuals, shape=(self.n_individuals, ).
        a: np.array, Archive of inferior solutions, shape=(n, self.ndim_problem).
    Returns:
        x_new: np.array, The new population after restart, shape=(self.n_individuals, self.ndim_problem).
        y_new: np.array, The fitness of new population after restart, shape=(self.n_individuals,).
        a_new: np.array, The archive of inferior solutions after restart, keep at least 50 records of individuals with the best fitness during each restart, shape=(n, self.ndim_problem).     
'''

func_template = '''thought:{...}
```python
Code:
def <method_name>(self, <method_args>):
    ...
```
'''