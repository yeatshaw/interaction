# Module Name: BP1DEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates constructive heuristic for 1-dimensional bin packing problem.
#              Given a set of bins and items, iteratively assign one item to feasible bins.
#              Design the optimal heuristic in each iteration to minimize the used bins.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# 
# Parameters:
#    -   n_bins: number of bins: int (default: 10).
#    -   n_instance: number of instances: int (default: 16).
#    -   n_items: number of items: int (default: 10).
#    -   bin_capacity: capacity of bins: int (default: 100).
#    -   timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 60).
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
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any, List, Tuple
import copy
import os
import pickle
from pathlib import Path

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bp_1d_construct.get_instance import GetData

__all__ = ['BP1DEvaluation']


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written with NumPy 2.x in older NumPy environments."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


class BP1DEvaluation(Evaluation):
    """Evaluator for the 1D Bin Packing Problem."""

    def __init__(self,
                 timeout_seconds: int = 60,
                 n_bins: int = 500,
                 n_instance: int = 8,
                 n_items: int = 500,
                 bin_capacity: int = 100,
                 dataset_path: str | Path | None = None,
                 **kwargs):
        """
        Args:
            n_bins: The number of available bins at the beginning.
        """
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)

        self.dataset_path = dataset_path or os.environ.get("LLM4AD_BP1D_TRAIN_DATA")
        self.n_instance = n_instance
        self.n_items = n_items
        self.bin_capacity = bin_capacity
        self.n_bins = n_bins
        if self.dataset_path:
            path = Path(self.dataset_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"BP_1d training dataset not found: {path}")
            with path.open("rb") as file:
                loaded = _NumpyCompatUnpickler(file).load()
            self._datasets = self._normalize_datasets(loaded)
            if not self._datasets:
                raise ValueError(f"BP_1d training dataset is empty: {path}")
            self.n_instance = len(self._datasets)
            self.n_items = len(self._datasets[0][0])
            self.bin_capacity = self._datasets[0][1]
            self.n_bins = max(self.n_bins, max(len(instance[0])
                                               for instance in self._datasets))
        else:
            getData = GetData(self.n_instance, self.n_items, self.bin_capacity)
            self._datasets = getData.generate_instances()

    @staticmethod
    def _normalize_datasets(loaded):
        """Normalize pickle layouts to (items, capacity, name, lower_bound)."""
        if isinstance(loaded, dict):
            for key in ("instances", "data", "datasets", "bp_1d"):
                if key in loaded:
                    loaded = loaded[key]
                    break
            else:
                # A mapping of instance names to instance values is also valid.
                loaded = list(loaded.values())
        if isinstance(loaded, np.ndarray):
            loaded = loaded.tolist()
        if not isinstance(loaded, (list, tuple)):
            raise ValueError("BP_1d dataset must be a list or dictionary of instances")

        datasets = []
        for index, instance in enumerate(loaded):
            if isinstance(instance, dict):
                items = instance.get("item_weights", instance.get("items"))
                capacity = instance.get("bin_capacity", instance.get("capacity", 100))
                name = instance.get("instance", instance.get("name", f"instance_{index + 1:03d}"))
                lower_bound = instance.get("theoretical_lower_bound")
            elif isinstance(instance, (list, tuple)) and len(instance) == 2:
                items, capacity = instance
                name = f"instance_{index + 1:03d}"
                lower_bound = None
            else:
                raise ValueError(f"Invalid BP_1d instance at index {index}")
            items = np.asarray(items).reshape(-1).tolist()
            if not items or any(float(item) <= 0 for item in items):
                raise ValueError(f"Invalid item weights at BP_1d instance {index}")
            item_weights = [int(item) for item in items]
            capacity = int(capacity)
            if capacity <= 0 or any(item <= 0 or item > capacity for item in item_weights):
                raise ValueError(f"Invalid BP_1d weights/capacity at instance {index}")
            if lower_bound is None:
                lower_bound = int(np.ceil(sum(item_weights) / capacity))
            datasets.append((item_weights, capacity, str(name), int(lower_bound)))
        return datasets

    def plot_bins(self, bins: List[List[int]], bin_capacity: int):
        """
        Plot the bins and their contents.

        Args:
            bins: A list of bins, where each bin is a list of item weights.
            bin_capacity: The capacity of each bin.
        """
        fig, ax = plt.subplots()

        # Create a bar plot for each bin
        for i, bin_content in enumerate(bins):
            # Calculate the cumulative sum of item weights for stacking
            cumulative_weights = [sum(bin_content[:j + 1]) for j in range(len(bin_content))]
            # Plot the bin as a bar, with items stacked
            ax.bar(i, cumulative_weights[-1] if cumulative_weights else 0, color='lightblue', edgecolor='black')
            # Plot individual items as stacked segments
            for j, weight in enumerate(bin_content):
                ax.bar(i, weight, bottom=cumulative_weights[j] - weight, edgecolor='black')

        # Set plot labels and title
        ax.set_xlabel('Bin Index')
        ax.set_ylabel('Weight')
        ax.set_title(f'1D Bin Packing Solution (Bin Capacity: {bin_capacity})')
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f'Bin {i + 1}' for i in range(len(bins))])
        ax.axhline(bin_capacity, color='red', linestyle='--', label='Bin Capacity')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

    def pack_items(self, item_weights: List[int], bin_capacity: int, eva: Callable, n_bins: int) -> Tuple[int, List[List[int]]]:
        """
        Pack items into bins using a constructive heuristic.

        Args:
            item_weights: A list of item weights.
            bin_capacity: The capacity of each bin.
            eva: The constructive heuristic function to select the next item and bin.
            n_bins: The number of available bins at the beginning.

        Returns:
            A tuple containing:
            - The total number of bins used.
            - A list of bins, where each bin is a list of item weights.
        """
        bins = [[] for _ in range(n_bins)]  # Initialize n_bins empty bins
        remaining_items = item_weights.copy()  # Copy of item weights to track remaining items
        remaining_capacities = [bin_capacity] * n_bins  # Initialize remaining capacities of all bins

        while remaining_items:
            # Use the heuristic to select the next item and bin
            remaining_items_copy = copy.deepcopy(remaining_items)
            remaining_capacities_copy = copy.deepcopy(remaining_capacities)
            try:
                selected_item, selected_bin = eva(
                    remaining_items_copy, remaining_capacities_copy)
            except (TypeError, ValueError):
                return None

            if selected_bin is None or selected_item not in remaining_items:
                return None
            if not isinstance(selected_bin, (int, np.integer)):
                return None
            if selected_bin < 0 or selected_bin >= n_bins:
                return None
            if selected_item > remaining_capacities[selected_bin]:
                return None

            bins[selected_bin].append(selected_item)
            remaining_capacities[selected_bin] -= selected_item

            if remaining_capacities[selected_bin] < 0:
                return None

            # Remove the selected item from the remaining items
            remaining_items.remove(selected_item)

        if len(remaining_items) > 0:
            return None

        # Calculate the number of bins used (bins that contain at least one item)
        used_bins = sum(1 for bin_content in bins if bin_content)

        return used_bins, bins

    def evaluate(self, eva: Callable) -> float:
        """
        Evaluate the constructive heuristic for the 1D Bin Packing Problem.

        Args:
            instance_data: List of tuples containing the item weights and bin capacity.
            n_ins: Number of instances to evaluate.
            eva: The constructive heuristic function to evaluate.
            n_bins: The number of available bins at the beginning.

        Returns:
            The average number of bins used across all instances.
        """
        total_bins = 0

        for instance in self._datasets:
            item_weights, bin_capacity = instance[:2]
            result = self.pack_items(item_weights, bin_capacity, eva, self.n_bins)
            if result is None:
                return None
            num_bins, _ = result
            total_bins += num_bins

        average_bins = total_bins / self.n_instance
        return -average_bins  # Negative because we want to minimize the number of bins

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)


if __name__ == '__main__':

    def determine_next_assignment(remaining_items: List[int], remaining_capacities: List[int]) -> Tuple[int, int | None]:
        """
        Determine the next item and bin to pack based on a greedy heuristic.

        Args:
            remaining_items: A list of remaining item weights.
            remaining_capacities: A list of remaining capacities of feasible bins.

        Returns:
            A tuple containing:
            - The selected item to pack.
            - The selected bin to pack the item into (or None if no feasible bin is found).
        """
        # Simple greedy heuristic: choose the largest item that fits into the bin with the smallest remaining capacity
        for item in sorted(remaining_items, reverse=True):  # Try largest items first
            for bin_id, capacity in enumerate(remaining_capacities):
                if item <= capacity:
                    return item, bin_id  # Return the selected item and bin
        return remaining_items[0], None  # If no feasible bin is found, return the first item and no bin


    bp1d = BP1DEvaluation()
    ave_bins = bp1d.evaluate_program('_', determine_next_assignment)
    print(ave_bins)
