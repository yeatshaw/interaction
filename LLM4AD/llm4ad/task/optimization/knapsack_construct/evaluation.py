# Module Name: KnapsackEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the Knapsack Problem.
#              Given a set of items with weights and values, the goal is to select a subset of items
#              that maximizes the total value while not exceeding the knapsack's capacity.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 20).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - n_items: Number of items available: int (default: 20).
#    - knapsack_capacity: Maximum capacity of the knapsack: int (default: 50).
#
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Any, List, Tuple
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.knapsack_construct.get_instance import GetData
from llm4ad.task.optimization.knapsack_construct.template import task_description

__all__ = ['KnapsackEvaluation']


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written with NumPy 2.x from older NumPy environments."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


class KnapsackEvaluation(Evaluation):
    """Evaluator for the Knapsack Problem."""

    def __init__(self,
                 timeout_seconds=20,
                 n_instance=32,
                 n_items=50,
                 knapsack_capacity=100,
                 dataset_path: str | Path | None = None,
                 **kwargs):
        """
        Initialize the evaluator for the Knapsack Problem.
        """
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)

        self.dataset_path = dataset_path or os.environ.get("LLM4AD_KP_TRAIN_DATA")
        self.n_instance = n_instance
        self.n_items = n_items
        self.knapsack_capacity = knapsack_capacity
        if self.dataset_path:
            path = Path(self.dataset_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Knapsack training dataset not found: {path}")
            with path.open("rb") as file:
                loaded = _NumpyCompatUnpickler(file).load()
            self._datasets = self._normalize_datasets(loaded)
            if not self._datasets:
                raise ValueError(f"Knapsack training dataset is empty: {path}")
            self.n_instance = len(self._datasets)
            self.n_items = len(self._datasets[0][0])
            self.knapsack_capacity = int(self._datasets[0][2])
        else:
            getData = GetData(self.n_instance, self.n_items, self.knapsack_capacity)
            self._datasets = getData.generate_instances()

    @staticmethod
    def _normalize_datasets(loaded):
        """Normalize pickle layouts to (item_weights, item_values, capacity)."""
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        if isinstance(loaded, dict):
            for key in ("instances", "data", "datasets", "knapsack", "kp"):
                if key in loaded:
                    loaded = loaded[key]
                    break
            else:
                loaded = list(loaded.values())
            if isinstance(loaded, dict):
                loaded = list(loaded.values())
        if isinstance(loaded, np.ndarray):
            loaded = loaded.tolist()
        if not isinstance(loaded, (list, tuple)):
            raise ValueError("Knapsack dataset must be a list or dictionary of instances")

        datasets = []
        for index, instance in enumerate(loaded):
            if isinstance(instance, dict):
                weights = instance.get("weights", instance.get("item_weights"))
                values = instance.get("values", instance.get("item_values"))
                capacity = instance.get("capacity", instance.get("knapsack_capacity"))
            elif isinstance(instance, (list, tuple)) and len(instance) >= 3:
                weights, values, capacity = instance[:3]
            else:
                raise ValueError(f"Invalid knapsack instance at index {index}")

            item_weights = [int(value) for value in np.asarray(weights).reshape(-1)]
            item_values = [float(value) for value in np.asarray(values).reshape(-1)]
            capacity = int(capacity)
            if capacity <= 0:
                raise ValueError(f"Invalid knapsack capacity at index {index}")
            if not item_weights or len(item_weights) != len(item_values):
                raise ValueError(f"Invalid knapsack item arrays at index {index}")
            if any(weight <= 0 for weight in item_weights):
                raise ValueError(f"Invalid knapsack weights at index {index}")
            datasets.append((item_weights, item_values, capacity))
        return datasets

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def plot_solution(self, item_weights: list, item_values: list, selected_indices: list, knapsack_capacity: int):
        """
        Plot the solution of the Knapsack problem.

        Args:
            item_weights: A list of item weights.
            item_values: A list of item values.
            selected_indices: A list of indices of selected items.
            knapsack_capacity: The capacity of the knapsack.
        """
        # Prepare data for plotting
        selected_weights = [item_weights[i] for i in selected_indices]
        selected_values = [item_values[i] for i in selected_indices]
        total_weight = sum(selected_weights)
        total_value = sum(selected_values)

        # Create a bar plot for selected items
        fig, ax = plt.subplots()
        x = range(len(selected_indices))
        ax.bar(x, selected_weights, label='Weight', color='blue', alpha=0.6)
        ax.bar(x, selected_values, label='Value', color='orange', alpha=0.6, bottom=selected_weights)

        # Add labels and title
        ax.set_xlabel('Selected Items')
        ax.set_ylabel('Weight / Value')
        ax.set_title(f'Knapsack Solution\nTotal Weight: {total_weight}/{knapsack_capacity}, Total Value: {total_value}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Item {i}' for i in selected_indices])
        ax.legend()

        plt.show()

    def pack_items(self, item_weights: List[int], item_values: List[int], knapsack_capacity: int, eva: Callable) -> Tuple[int, List[int]]:
        """
        Select items for the knapsack using a constructive heuristic.

        Args:
            item_weights: A list of item weights.
            item_values: A list of item values.
            knapsack_capacity: The capacity of the knapsack.
            eva: The constructive heuristic function to select the next item.

        Returns:
            A tuple containing:
            - The total value of the selected items.
            - A list of selected item indices.
        """
        remaining_items = list(zip(item_weights, item_values, range(len(item_weights))))  # Track weights, values, and indices
        selected_items = []  # List of selected item indices
        remaining_capacity = knapsack_capacity  # Track remaining capacity
        total_value = 0  # Track total value of selected items

        while remaining_items and remaining_capacity > 0:
            # Use the heuristic to select the next item
            selected_item = eva(remaining_capacity, remaining_items)

            if selected_item is not None:
                weight, value, index = selected_item
                if weight <= remaining_capacity:
                    # Add the selected item to the knapsack
                    selected_items.append(index)
                    total_value += value
                    remaining_capacity -= weight
                # Remove the selected item from the remaining items
                remaining_items.remove(selected_item)
            else:
                break

        return total_value, selected_items

    def evaluate(self, eva: Callable) -> float:
        """
        Evaluate the constructive heuristic for the Knapsack Problem.

        Args:
            instance_data: List of tuples containing the item weights, values, and knapsack capacity.
            n_ins: Number of instances to evaluate.
            eva: The constructive heuristic function to evaluate.

        Returns:
            The average total value of selected items across all instances.
        """
        total_value = 0

        for instance in self._datasets[:self.n_instance]:
            item_weights, item_values, knapsack_capacity = instance
            value, _ = self.pack_items(item_weights, item_values, knapsack_capacity, eva)
            total_value += value

        average_value = total_value / self.n_instance
        # Recipe-MCTS/EoH retain larger scores, and KP maximizes total value.
        return average_value


if __name__ == '__main__':

    def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
        """
        Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

        Args:
            remaining_capacity: The remaining capacity of the knapsack.
            remaining_items: List of tuples containing (weight, value, index) of remaining items.

        Returns:
            The selected item as a tuple (weight, value, index), or None if no item fits.
        """
        best_item = None
        best_ratio = -1  # Initialize with a negative value to ensure any item will have a higher ratio

        for item in remaining_items:
            weight, value, index = item
            if weight <= remaining_capacity:
                ratio = value / weight  # Calculate value-to-weight ratio
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_item = item

        return best_item


    bp1d = KnapsackEvaluation()
    ave_bins = bp1d.evaluate_program('_', select_next_item)
    print(ave_bins)
