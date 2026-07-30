import numpy as np  # engine for numerical computing
from scipy.stats import cauchy  # Cauchy continuous random variable

from .jade import JADE  # adaptive differential evolution (JADE)
from .de import DE

class LSHADE(JADE):
    """Success-History based Adaptive Differential Evolution (SHADE).

        Parameters
        ----------
        problem : `dict`
                  problem arguments with the following common settings (`keys`):
                    * 'fitness_function' - objective function to be **minimized** (`func`),
                    * 'ndim_problem'     - number of dimensionality (`int`),
                    * 'upper_boundary'   - upper boundary of search range (`array_like`),
                    * 'lower_boundary'   - lower boundary of search range (`array_like`).
        options : `dict`
                  optimizer options with the following common settings (`keys`):
                    * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                    * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                    * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
                  and with the following particular settings (`keys`):
                    * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `100`),
                    * 'mu'            - mean of normal distribution for adaptation of crossover probability (`float`,
                      default: `0.5`),
                    * 'median'        - median of Cauchy distribution for adaptation of mutation factor (`float`,
                      default: `0.5`),
                    *  'h'            - length of historical memory (`int`, default: `100`).

        Examples
        --------
        Use the optimizer to minimize the well-known test function
        `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`

        For its correctness checking of coding, refer to `this code-based repeatability report
        <https://tinyurl.com/vm3w7se4>`_ for more details.

        Attributes
        ----------
        h             : `int`
                        length of historical memory.
        median        : `float`
                        median of Cauchy distribution for adaptation of mutation factor.
        mu            : `float`
                        mean of normal distribution for adaptation of crossover probability.
        n_individuals : `int`
                        number of offspring, aka offspring population size.

        References
        ----------
        Tanabe, R. and Fukunaga, A., 2013, June.
        `Success-history based parameter adaptation for differential evolution.
        <https://ieeexplore.ieee.org/document/6557555>`_
        In IEEE Congress on Evolutionary Computation (pp. 71-78). IEEE.
        """
    def __init__(self, problem, options, rand_seed, optimal_value):
        JADE.__init__(self, problem, options)
        self.h = 100  # length of historical memory
        assert 0 < self.h
        self.m_mu = np.ones(self.h)*self.mu  # means of normal distribution
        self.m_median = np.ones(self.h)*self.median  # medians of Cauchy distribution
        self._k = 0  # index to update
        self.p_min = 2.0/self.n_individuals
        self.initial_pop_size = self.n_individuals

        # set seed
        self.rng_initialization = np.random.default_rng(rand_seed)
        self.rng_optimization = np.random.default_rng(rand_seed)

        # restart
        self.stagnation_threshold = options.get('stagnation_threshold', 20)  # 连续多少代无改善视为停滞
        self.stagnation_tol = options.get('stagnation_tol', 1e-8)  # 相对改善阈值
        self._stagnation_counter = 0
        self._best_y_prev = 1e+8

    def restart(self, x=None, y=None, a=None, args=None):
        x_new = x.copy()
        y_new = y.copy()
        a_new = a.copy() if a is not None and len(a) > 0 else np.empty((0, self.ndim_problem))

        best_y_current = np.min(y_new)

        if self._best_y_prev is not None:
            if self._best_y_prev == 0.0:
                improved = best_y_current < self._best_y_prev - self.stagnation_tol
            else:
                improved = (self._best_y_prev - best_y_current) / (
                            np.abs(self._best_y_prev) + 1e-30) > self.stagnation_tol
            if improved:
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0

        self._best_y_prev = best_y_current

        if self._stagnation_counter < self.stagnation_threshold:
            return x_new, y_new, a_new

        self._stagnation_counter = 0

        n = self.n_individuals
        d = self.ndim_problem
        lb = self.lower_boundary
        ub = self.upper_boundary
        rng = self.rng_optimization

        sorted_indices = np.argsort(y_new)
        n_elite = max(1, int(np.ceil(self.p_min * n)))
        n_opp = int(0.7 * n)

        # Opposition for worst 70%
        worst_indices = sorted_indices[-n_opp:]
        opp_x = lb + ub - x_new[worst_indices]
        opp_x = np.clip(opp_x, lb, ub)
        opp_y = np.array([self._evaluate_fitness(opp_x[i:i + 1]) for i in range(n_opp)])

        # Pairwise selection
        for i, idx in enumerate(worst_indices):
            if opp_y[i] < y_new[idx]:
                x_new[idx] = opp_x[i]
                y_new[idx] = opp_y[i]

        sorted_indices = np.argsort(y_new)
        elites_x = x_new[sorted_indices[:n_elite]].copy()
        elites_y = y_new[sorted_indices[:n_elite]].copy()

        # Archive pruning to top-30 by fitness
        if a_new.shape[0] > 30:
            a_y = np.array([self._evaluate_fitness(a_new[i:i + 1]) for i in range(a_new.shape[0])])
            a_new = a_new[np.argsort(a_y)[:30]]

        # DE/rand/1 with archive
        n_fill = n - n_elite
        fill_x = np.empty((n_fill, d))
        fill_y = np.empty(n_fill)
        cr = 0.9
        f = 0.5

        for i in range(n_fill):
            r1, r2 = rng.choice(n, 2, replace=False)
            if a_new.shape[0] > 0 and rng.random() < 0.5:
                r3 = a_new[rng.integers(0, a_new.shape[0])]
            else:
                r3 = x_new[rng.integers(0, n)]

            mutant = x_new[r1] + f * (x_new[r2] - r3)
            mutant = np.clip(mutant, lb, ub)

            trial = x_new[rng.integers(0, n)].copy()
            j_rand = rng.integers(0, d)
            for j in range(d):
                if rng.random() < cr or j == j_rand:
                    trial[j] = mutant[j]

            trial = np.clip(trial, lb, ub)
            fill_x[i] = trial
            fill_y[i] = self._evaluate_fitness(trial.reshape(1, d))

        x_new = np.vstack([elites_x, fill_x])
        y_new = np.concatenate([elites_y, fill_y])

        self.m_median[:] = 0.5
        self._best_y_prev = np.min(y_new)

        return x_new, y_new, a_new
    
    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))  # mutated population
        f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
        #x_un = np.vstack((np.copy(x), a))  # union of population x and archive a
        x_un = np.copy(x)
        r = self.rng_optimization.choice(self.h, (self.n_individuals,))
        order = np.argsort(y)[:]
        p = (0.2 - self.p_min)*self.rng_optimization.random((self.n_individuals,)) + self.p_min
        idx = [order[self.rng_optimization.choice(int(i))] for i in np.ceil(p*self.n_individuals)]
        for k in range(self.n_individuals):
            #f_mu = np.full((self.n_individuals,), 0.5)
            f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice([i for i in range(self.n_individuals) if i != k])
            r2 = self.rng_optimization.choice([i for i in range(len(x_un)) if i != k and i != r1])
            x_mu[k] = x[k] + f_mu[k]*(x[idx[k]] - x[k]) + f_mu[k]*(x[r1] - x_un[r2])
        return x_mu, f_mu, r

    def crossover(self, x_mu=None, x=None, r=None):
        import numpy as np
        from scipy.spatial.distance import cdist
    
        x_cr = np.copy(x_mu)
        p_cr = np.zeros(self.n_individuals)
    
        bracket_size = max(4, self.n_individuals // 4)
        n_brackets = max(2, self.n_individuals // bracket_size)
    
        fitness_ranks = np.argsort(np.argsort(r))
        centroid = np.mean(x_mu, axis=0)
        distances = np.linalg.norm(x_mu - centroid, axis=1)
        distance_ranks = np.argsort(np.argsort(distances))
    
        total_generations = self.max_function_evaluations / self.initial_pop_size
        cycle_period = total_generations * 0.2
        phase = (self._n_generations % cycle_period) / cycle_period
    
        oscillation_factor = 0.5 * (1 + np.sin(2 * np.pi * phase))
        exploration_weight = 0.8 * oscillation_factor + 0.2 * (1 - oscillation_factor)
    
        combined_scores = (1 - exploration_weight) * fitness_ranks + exploration_weight * distance_ranks
        sorted_indices = np.argsort(combined_scores)
    
        brackets = [[] for _ in range(n_brackets)]
        for i, idx in enumerate(sorted_indices):
            bracket_idx = i % n_brackets if (i // n_brackets) % 2 == 0 else n_brackets - 1 - (i % n_brackets)
            brackets[bracket_idx].append(idx)
    
        momentum = 0.5 * (1 + np.tanh((self._n_generations - total_generations * 0.3) / (total_generations * 0.1)))
    
        for bracket_idx, bracket in enumerate(brackets):
            if len(bracket) < 2:
                continue
    
            bracket_fitness = r[bracket]
            competitiveness = np.std(bracket_fitness) / (np.mean(bracket_fitness) + 1e-8)
    
            bracket_ranks = np.argsort(bracket_fitness)
            winners = [bracket[i] for i in bracket_ranks[:len(bracket)//2]]
            losers = [bracket[i] for i in bracket_ranks[len(bracket)//2:]]
    
            for i in range(len(bracket)):
                individual_idx = bracket[i]
    
                if individual_idx in winners:
                    partner_pool = [w for w in winners if w != individual_idx]
                    base_prob = 0.15 + 0.4 * (1 - momentum)
                else:
                    partner_pool = winners
                    base_prob = 0.65 + 0.15 * momentum
    
                if partner_pool:
                    partner = self.rng_optimization.choice(partner_pool)
                else:
                    partner = self.rng_optimization.choice([idx for idx in bracket if idx != individual_idx])
    
                oscillation_bonus = 0.1 * np.sin(4 * np.pi * phase)
                competitiveness_factor = np.tanh(2 * competitiveness)
                p_cr[individual_idx] = base_prob * (0.85 + 0.3 * competitiveness_factor) + oscillation_bonus
                p_cr[individual_idx] *= (0.75 + 0.25 * self.rng_optimization.random())
                p_cr[individual_idx] = np.clip(p_cr[individual_idx], 0.1, 0.9)
    
                mask = self.rng_optimization.random(self.ndim_problem) < p_cr[individual_idx]
                if not np.any(mask):
                    mask[self.rng_optimization.integers(0, self.ndim_problem)] = True
    
                x_cr[individual_idx][mask] = x[partner][mask]
    
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        # set successful mutation factors, crossover probabilities, fitness differences
        f, p, d = np.empty((0,)), np.empty((0,)), np.empty((0,))
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive of inferior solutions
                f = np.hstack((f, f_mu[k]))  # archive of successful mutation factors
                p = np.hstack((p, p_cr[k]))  # archive of successful crossover probabilities
                d = np.hstack((d, y[k] - yy))  # archive of successful fitness differences
                x[k], y[k] = x_cr[k], yy
        if (len(p) != 0) and (len(f) != 0):
            w = d/np.sum(d)
            #self.m_mu[self._k] = np.sum(w*p)  # for normal distribution
            self.m_median[self._k] = np.sum(w*np.power(f, 2))/np.sum(w*f)  # for Cauchy distribution
            self._k = (self._k + 1) % self.h
        return x, y, a

    def change_population(self, x=None, y=None, a=None, args=None):
        """
        max_iterations = max(2, self.max_function_evaluations // self.initial_pop_size)  # Ensure at least 2 iterations
        reduction_factor = (self.initial_pop_size - 4) / (max_iterations - 1)
        self.n_individuals = max(4, int(self.initial_pop_size - self._n_generations * reduction_factor))

        # Select the best individuals to form the new population
        if len(a) > self.n_individuals:
            indices = np.argsort(y)[:self.n_individuals]
            x = x[indices]
            y = y[indices]
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        else:
            # If the archive size is less than the new population size, keep it as is
            pass
        """
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu, r = self.mutate(x.copy(), y.copy(), a.copy())
        x_cr, p_cr = self.crossover(x_mu.copy(), x.copy(), r.copy())
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        x, y, a = self.change_population(x.copy(), y.copy(), a.copy())
        # if len(a) > self.n_individuals:  # randomly remove solutions to keep archive size fixed
        #     a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        self._n_generations += 1
        
        x, y, a = self.restart(x, y, a)
        return x, y, a
