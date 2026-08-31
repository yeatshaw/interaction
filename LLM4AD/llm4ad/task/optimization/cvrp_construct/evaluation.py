from __future__ import annotations

from typing import Any
import numpy as np

from llm4ad.base import Evaluation
from .get_instance import GetData


class CVRPEvaluation(Evaluation):
    """Evaluate source-code CVRP construction heuristics on random instances."""

    def __init__(self, timeout_seconds=30, problem_size=100, n_instance=50,
                 capacity=40, **kwargs):
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)
        self.problem_size = problem_size
        self.n_instance = n_instance
        self.capacity = capacity
        self._datasets = GetData(n_instance, problem_size + 1, capacity).generate_instances()
        for _, _, demands, _ in self._datasets:
            demands[0] = 0

    @staticmethod
    def _load_heuristic(function_source):
        namespace = {'np': np}
        exec(function_source, namespace)
        heuristic = namespace.get('select_next_node')
        if not callable(heuristic):
            raise ValueError('Program does not define callable select_next_node.')
        return heuristic

    @staticmethod
    def _evaluate_data(heuristic, data):
        coordinates, distance_matrix, demands, capacity = data
        customer_count = len(coordinates) - 1
        route, current_node, current_load = [0], 0, 0.0
        unvisited = set(range(1, customer_count + 1))
        while unvisited:
            feasible = np.asarray([node for node in sorted(unvisited)
                                   if current_load + demands[node] <= capacity], dtype=int)
            if feasible.size == 0:
                if current_node != 0:
                    route.append(0)
                current_node, current_load = 0, 0.0
                continue
            next_node = int(heuristic(
                current_node, 0, feasible.copy(), capacity - current_load,
                demands.copy(), distance_matrix.copy()))
            if next_node == 0:
                if current_node == 0:
                    return None
                route.append(0)
                current_node, current_load = 0, 0.0
                continue
            if next_node not in unvisited or next_node not in feasible:
                return None
            route.append(next_node)
            current_load += float(demands[next_node])
            current_node = next_node
            unvisited.remove(next_node)
        if route[-1] != 0:
            route.append(0)
        return float(sum(distance_matrix[a, b] for a, b in zip(route, route[1:])))

    def evaluate_instance(self, function_source, data):
        return self._evaluate_data(self._load_heuristic(function_source), data)

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(str(program_str))

    def evaluate(self, program_source):
        heuristic = self._load_heuristic(program_source)
        costs = [self._evaluate_data(heuristic, data)
                 for data in self._datasets[:self.n_instance]]
        if any(cost is None or not np.isfinite(cost) for cost in costs):
            return None
        return -float(np.mean(costs))
