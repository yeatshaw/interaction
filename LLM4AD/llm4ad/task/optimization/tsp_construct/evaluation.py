from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from .get_instance import GetData

__all__ = ["TSPEvaluation", "RefineEVOTSPEvaluation"]


class TSPEvaluation(Evaluation):
    """Evaluate constructive TSP heuristics by maximizing negative distance."""

    def __init__(self, timeout_seconds=30, n_instance=50, problem_size=100,
                 dataset_path: str | Path | None = None, seed=2024, **kwargs):
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)
        strict_recipe_dataset_format = bool(
            kwargs.pop("strict_recipe_dataset_format", False))
        dataset_path = dataset_path or os.environ.get("LLM4AD_TSP_TRAIN_DATA")
        if dataset_path:
            with Path(dataset_path).expanduser().open("rb") as file:
                datasets = pickle.load(file)
        else:
            datasets = GetData(n_instance, problem_size, seed=seed).generate_instances()

        validator = (self._validate_recipe_datasets
                     if strict_recipe_dataset_format
                     else self._validate_datasets)
        self._datasets = validator(datasets)
        self.n_instance = len(self._datasets)
        self.problem_size = len(self._datasets[0][0])

    @staticmethod
    def _validate_recipe_datasets(datasets):
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError("TSP dataset must be a non-empty list of instance pairs.")
        validated = []
        for index, item in enumerate(datasets):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"TSP instance {index} must be (coordinates, distance_matrix).")
            coordinates = np.asarray(item[0], dtype=float)
            distance_matrix = np.asarray(item[1], dtype=float)
            city_count = len(coordinates)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError(f"TSP instance {index} coordinates must have shape [n, 2].")
            if city_count < 2 or distance_matrix.shape != (city_count, city_count):
                raise ValueError(
                    f"TSP instance {index} has an invalid distance matrix shape.")
            if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(distance_matrix)):
                raise ValueError(f"TSP instance {index} contains non-finite values.")
            validated.append((coordinates, distance_matrix))
        return validated

    @staticmethod
    def _validate_datasets(datasets):
        # Training data is commonly stored as a list of
        # ``(coordinates, distance_matrix)`` pairs.  TSPLIB data is stored as
        # ``{instance_name: {coordinates, optimal_value, ...}}``; normalize
        # both forms here so the evaluator does not need task-specific callers.
        if isinstance(datasets, dict):
            normalized = []
            for name, value in datasets.items():
                if not isinstance(value, dict) or "coordinates" not in value:
                    raise ValueError(
                        f"TSP instance {name!r} must contain 'coordinates'.")
                coordinates = np.asarray(value["coordinates"], dtype=float)
                distance_matrix = value.get("distance_matrix")
                if distance_matrix is None:
                    distance_matrix = value.get("distances")
                if distance_matrix is None:
                    if len(coordinates) > 2000:
                        raise ValueError(
                            f"TSP instance {name!r} has {len(coordinates)} nodes; "
                            "provide a precomputed distance_matrix or filter "
                            "large instances before evaluation.")
                    delta = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
                    distance_matrix = np.linalg.norm(delta, axis=2)
                normalized.append((coordinates, np.asarray(distance_matrix, dtype=float)))
            datasets = normalized
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError(
                "TSP dataset must be a non-empty list or instance dictionary.")
        validated = []
        for index, item in enumerate(datasets):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"TSP instance {index} must be (coordinates, distance_matrix).")
            coordinates = np.asarray(item[0], dtype=float)
            distance_matrix = np.asarray(item[1], dtype=float)
            city_count = len(coordinates)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError(f"TSP instance {index} coordinates must have shape [n, 2].")
            if city_count < 2 or distance_matrix.shape != (city_count, city_count):
                raise ValueError(
                    f"TSP instance {index} has an invalid distance matrix shape.")
            if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(distance_matrix)):
                raise ValueError(f"TSP instance {index} contains non-finite values.")
            validated.append((coordinates, distance_matrix))
        return validated

    @staticmethod
    def _load_heuristic(function_source):
        namespace = {"np": np}
        exec(function_source, namespace)
        heuristic = namespace.get("select_next_node")
        if not callable(heuristic):
            raise ValueError("Program does not define callable select_next_node.")
        return heuristic

    @staticmethod
    def _evaluate_data(heuristic, data):
        coordinates, distance_matrix = data
        city_count = len(coordinates)
        route = [0]
        unvisited = set(range(1, city_count))
        current_node = 0

        while unvisited:
            candidates = np.asarray(sorted(unvisited), dtype=int)
            try:
                # Invalid generated heuristics often take the mean of an empty
                # filtered set. Convert that numerical warning into an
                # infeasible evaluation instead of allowing NaN to propagate.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    next_node = int(heuristic(
                        current_node, 0, candidates.copy(),
                        distance_matrix.copy()))
            except (TypeError, ValueError, IndexError, OverflowError,
                    ZeroDivisionError, FloatingPointError, RuntimeWarning):
                return None
            if next_node not in unvisited:
                return None
            route.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        route.append(0)
        distance = sum(distance_matrix[a, b] for a, b in zip(route, route[1:]))
        return float(distance) if np.isfinite(distance) else None

    def evaluate_instance(self, function_source, data):
        """Return one instance's route distance for final test reporting."""
        return self._evaluate_data(self._load_heuristic(function_source), data)

    def evaluate_program(self, program_str: str,
                         callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, heuristic):
        if not callable(heuristic):
            heuristic = self._load_heuristic(str(heuristic))
        distances = [self._evaluate_data(heuristic, data)
                     for data in self._datasets]
        if not distances or any(value is None for value in distances):
            return None
        return -float(np.mean(distances))


class RefineEVOTSPEvaluation(TSPEvaluation):
    """RefineEvo-compatible TSP evaluation with the LLM4AD score direction.

    RefineEvo rebuilds the Euclidean distance matrix from coordinates for each
    instance and sums the closed tour length.  This adapter keeps that route
    evaluation unchanged, but returns ``-mean_distance`` so MCTS/EoH can still
    maximize the score.  ``TSPEvaluation`` remains available for the original
    stored-distance-matrix behavior.
    """

    @staticmethod
    def _evaluate_data(heuristic, data):
        from scipy.spatial import distance_matrix

        coordinates, _ = data
        coordinates = np.asarray(coordinates, dtype=float)
        dist_mat = distance_matrix(coordinates, coordinates)
        problem_size = coordinates.shape[0]
        start_node = 0
        solution = [start_node]
        unvisited = set(range(problem_size))
        unvisited.remove(start_node)

        for _ in range(problem_size - 1):
            try:
                next_node = int(heuristic(
                    current_node=solution[-1],
                    destination_node=start_node,
                    unvisited_nodes=set(unvisited),
                    distance_matrix=dist_mat.copy(),
                ))
            except (TypeError, ValueError, IndexError, OverflowError,
                    ZeroDivisionError, FloatingPointError, RuntimeWarning):
                return None
            if next_node not in unvisited:
                return None
            solution.append(next_node)
            unvisited.remove(next_node)

        return_node_distance = dist_mat[solution[-1], start_node]
        distance = sum(
            dist_mat[solution[index], solution[index + 1]]
            for index in range(len(solution) - 1)
        ) + return_node_distance
        return float(distance) if np.isfinite(distance) else None
