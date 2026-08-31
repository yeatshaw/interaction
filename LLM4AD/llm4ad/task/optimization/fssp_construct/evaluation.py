from __future__ import annotations

from typing import Any
import numpy as np

from llm4ad.base import Evaluation


class FSSPEvaluation(Evaluation):
    """Evaluate source-code permutation flow-shop construction heuristics."""

    def __init__(self, timeout_seconds=30, n_instance=50, n_jobs=50,
                 n_machines=20, seed=2024, **kwargs):
        super().__init__(use_numba_accelerate=False,
                         timeout_seconds=timeout_seconds)
        self.n_instance = n_instance
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        rng = np.random.default_rng(seed)
        self._datasets = [rng.integers(1, 100, size=(n_machines, n_jobs), dtype=np.int64)
                          for _ in range(n_instance)]

    @staticmethod
    def _load_heuristic(function_source):
        namespace = {'np': np}
        exec(function_source, namespace)
        heuristic = namespace.get('select_next_job')
        if not callable(heuristic):
            raise ValueError('Program does not define callable select_next_job.')
        return heuristic

    @staticmethod
    def _completion_times(processing_times, job_order):
        machine_completion = np.zeros(processing_times.shape[0], dtype=float)
        for job in job_order:
            previous_machine = 0.0
            for machine in range(processing_times.shape[0]):
                end = max(machine_completion[machine], previous_machine) + processing_times[machine, job]
                machine_completion[machine] = end
                previous_machine = end
        return machine_completion

    @classmethod
    def _evaluate_data(cls, heuristic, processing_times):
        job_count = processing_times.shape[1]
        order = []
        remaining = set(range(job_count))
        while remaining:
            machine_completion = cls._completion_times(processing_times, order)
            candidates = np.asarray(sorted(remaining), dtype=int)
            selected = int(heuristic(
                np.asarray(order, dtype=int), candidates.copy(),
                machine_completion.copy(), processing_times.copy()))
            if selected not in remaining:
                return None
            order.append(selected)
            remaining.remove(selected)
        return float(cls._completion_times(processing_times, order)[-1])

    def evaluate_instance(self, function_source, processing_times):
        return self._evaluate_data(self._load_heuristic(function_source), processing_times)

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(str(program_str))

    def evaluate(self, program_source):
        heuristic = self._load_heuristic(program_source)
        makespans = [self._evaluate_data(heuristic, data)
                     for data in self._datasets[:self.n_instance]]
        if any(value is None or not np.isfinite(value) for value in makespans):
            return None
        return -float(np.mean(makespans))
