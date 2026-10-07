from __future__ import annotations

import random
from numbers import Integral
from typing import Any

from llm4ad.base import Evaluation
from .template import task_description, template_program

__all__ = ["ScheduleInitEvaluation"]


class ScheduleInitEvaluation(Evaluation):
    """Evaluate initial circular-grid layouts using the PDF uniformity score."""

    UNIFORMITY_WEIGHT = 0.55

    def __init__(self, timeout_seconds=20, n_instances=32, seed=2024,
                 instances=None, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        if instances is None:
            if isinstance(n_instances, bool) or not isinstance(n_instances, int) or n_instances < 1:
                raise ValueError("n_instances must be a positive integer.")
            self._instances = self._generate_instances(n_instances, seed)
        else:
            self._instances = self._normalize_instances(instances)
        self.n_instances = len(self._instances)

    @staticmethod
    def _normalize_instance(instance, index):
        if not isinstance(instance, (tuple, list)) or len(instance) != 2:
            raise ValueError(f"Instance {index} must be (N, service_list).")
        n, service_list = instance
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1 or n > 10000:
            raise ValueError(f"Instance {index} has invalid N; expected 1 <= N <= 10000.")
        if not isinstance(service_list, (tuple, list)) or not 1 <= len(service_list) <= 100:
            raise ValueError(f"Instance {index} must contain 1 to 100 services.")

        normalized = []
        seen_ids = set()
        total = 0
        for pair in service_list:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Instance {index} contains an invalid service pair.")
            service_id, count = pair
            if (isinstance(service_id, bool) or not isinstance(service_id, Integral)
                    or service_id <= 0):
                raise ValueError(f"Instance {index} contains a non-positive service ID.")
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError(f"Instance {index} contains a non-positive service count.")
            service_id = int(service_id)
            count = int(count)
            if service_id in seen_ids:
                raise ValueError(f"Instance {index} contains duplicate service IDs.")
            seen_ids.add(service_id)
            normalized.append((service_id, count))
            total += count
        if total > n:
            raise ValueError(f"Instance {index} has more service granules than slots.")
        return int(n), tuple(normalized)

    @classmethod
    def _normalize_instances(cls, instances):
        if not isinstance(instances, (tuple, list)) or not instances:
            raise ValueError("instances must be a non-empty sequence of cases.")
        return tuple(cls._normalize_instance(item, index)
                     for index, item in enumerate(instances))

    @staticmethod
    def _generate_instances(n_instances, seed):
        rng = random.Random(seed)
        instances = []
        for index in range(n_instances):
            size_class = index % 5
            if size_class == 0:
                n = 1
            elif size_class == 1:
                n = rng.randint(2, 20)
            elif size_class == 2:
                n = rng.randint(21, 500)
            elif size_class == 3:
                n = rng.randint(501, 3000)
            else:
                n = rng.randint(3001, 10000)

            service_count = rng.randint(1, min(100, n))
            occupancy_mode = index % 4
            if occupancy_mode == 0:
                occupied = n
            elif occupancy_mode == 1:
                empty = rng.randint(1, max(1, n // 4))
                occupied = max(service_count, n - empty)
            elif occupancy_mode == 2:
                occupied = rng.randint(service_count, n)
            else:
                empty = rng.randint(0, n // 2)
                occupied = max(service_count, n - empty)

            counts = [1 for _ in range(service_count)]
            remaining = occupied - service_count
            count_mode = index % 3
            if count_mode == 0:
                while remaining > 0:
                    counts[rng.randrange(service_count)] += 1
                    remaining -= 1
            elif count_mode == 1:
                counts[0] += remaining
            else:
                position = 0
                while remaining > 0:
                    counts[position] += 1
                    remaining -= 1
                    position += 1
                    if position >= service_count:
                        position = 0

            service_ids = rng.sample(range(1, 2147483648), service_count)
            services = tuple(zip(service_ids, counts))
            instances.append((n, services))
        return tuple(instances)

    @staticmethod
    def score_grid(grid, n, service_list) -> float | None:
        """Return the PDF layout-uniformity score, or None for an invalid grid."""
        try:
            values = list(grid)
        except (TypeError, ValueError):
            return None
        if len(values) != n:
            return None

        expected = {int(service_id): int(count)
                    for service_id, count in service_list}
        expected[0] = n - sum(expected.values())
        positions = {}
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Integral):
                return None
            value = int(value)
            if value not in expected:
                return None
            positions.setdefault(value, []).append(index)

        for service_id, expected_count in expected.items():
            if len(positions.get(service_id, ())) != expected_count:
                return None

        uniformity = 0.0
        for service_id, locations in positions.items():
            count = len(locations)
            if count <= 1 or count == n:
                type_score = 1.0
            else:
                ideal_gap = n / count
                deviation = 0.0
                for index, location in enumerate(locations):
                    if index + 1 < count:
                        gap = locations[index + 1] - location
                    else:
                        gap = locations[0] + n - location
                    deviation += abs(gap - ideal_gap)
                worst_deviation = 2 * (count - 1) * (ideal_gap - 1)
                if worst_deviation <= 0:
                    return None
                type_score = 1.0 - deviation / worst_deviation
                type_score = max(0.0, min(1.0, type_score))
            uniformity += (count / n) * type_score
        return uniformity

    def evaluate_program(self, program_str: str,
                         callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, initializer: callable) -> float | None:
        if not callable(initializer):
            return None
        uniformity_scores = []
        for n, services in self._instances:
            try:
                grid = initializer(n, list(services))
            except Exception:
                return None
            uniformity = self.score_grid(grid, n, services)
            if uniformity is None:
                return None
            uniformity_scores.append(uniformity)
        mean_uniformity = sum(uniformity_scores) / len(uniformity_scores)
        return self.UNIFORMITY_WEIGHT * mean_uniformity
