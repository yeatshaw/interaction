from __future__ import annotations

from typing import Any
from pathlib import Path
import os
import pickle
import numpy as np

from llm4ad.base import Evaluation
from .get_instance import GetData


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written with NumPy 2.x from older NumPy environments."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


class CVRPEvaluation(Evaluation):
    """Evaluate source-code CVRP construction heuristics on random instances."""

    def __init__(self, timeout_seconds=30, problem_size=100, n_instance=50,
                 capacity=40, dataset_path: str | Path | None = None, **kwargs):
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)
        self.dataset_path = dataset_path or os.environ.get("LLM4AD_CVRP_TRAIN_DATA")
        self.problem_size = problem_size
        self.n_instance = n_instance
        self.capacity = capacity
        if self.dataset_path:
            path = Path(self.dataset_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"CVRP training dataset not found: {path}")
            with path.open("rb") as file:
                payload = _NumpyCompatUnpickler(file).load()
            self._datasets = self._normalize_datasets(payload)
            if not self._datasets:
                raise ValueError(f"CVRP training dataset is empty: {path}")
            self.n_instance = len(self._datasets)
            self.problem_size = len(self._datasets[0][0]) - 1
        else:
            self._datasets = GetData(
                n_instance, problem_size + 1, capacity).generate_instances()
        for _, _, demands, _ in self._datasets:
            demands[0] = 0

    @staticmethod
    def _normalize_datasets(payload):
        """Normalize common CVRP pickle layouts to evaluator tuples."""
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if isinstance(payload, dict):
            for wrapper in ("cvrp_dict", "datasets", "data", "instances"):
                if wrapper in payload and isinstance(
                        payload[wrapper], (dict, list, tuple, np.ndarray)):
                    payload = payload[wrapper]
                    break
            if isinstance(payload, dict):
                payload = list(payload.values())
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()
        if not isinstance(payload, (list, tuple)):
            raise ValueError("CVRP dataset must be a list or dictionary")

        datasets = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                coords = item.get("coordinates", item.get("coords"))
                matrix = item.get("distance_matrix", item.get("distances"))
                demands = item.get("demands")
                capacity = item.get("capacity")
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                coords, demands, capacity = item
                matrix = None
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                coords, matrix, demands, capacity = item
            elif (isinstance(item, (list, tuple)) and len(item) >= 5
                  and np.isscalar(item[0]) and np.isscalar(item[1])):
                # CVRPLIB layout: capacity, node_count, coords, demands, BKS.
                capacity, _, coords, demands = item[:4]
                matrix = None
            else:
                raise ValueError(f"Invalid CVRP instance at index {index}")

            coords = np.asarray(coords, dtype=float)
            demands = np.asarray(demands, dtype=float).copy()
            if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 2:
                raise ValueError(f"Invalid CVRP coordinates at index {index}")
            coords = coords[:, :2]
            if demands.shape != (len(coords),):
                raise ValueError(f"Invalid CVRP demands at index {index}")
            if matrix is None:
                delta = coords[:, None, :] - coords[None, :, :]
                matrix = np.sqrt(np.sum(delta * delta, axis=2))
            else:
                matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (len(coords), len(coords)):
                raise ValueError(f"Invalid CVRP distance matrix at index {index}")
            if capacity is None or float(capacity) <= 0:
                raise ValueError(f"Invalid CVRP capacity at index {index}")
            datasets.append((coords, matrix, demands, float(capacity)))
        return datasets

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
        if not costs:
            return None
        if any(cost is None or not np.isfinite(cost) for cost in costs):
            return None
        return -float(np.mean(costs))
