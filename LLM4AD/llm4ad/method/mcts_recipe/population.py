from __future__ import annotations

import copy
import math
import numpy as np


class RecipePopulation:
    """A private population snapshot for one tree node."""

    def __init__(self, individuals, pop_size, generation=0):
        self.individuals = list(individuals)
        self.pop_size = int(pop_size)
        self.generation = int(generation)

    @property
    def population(self):
        """Expose the collection name expected by the existing EoH prompts."""
        return self.individuals

    def clone(self):
        return RecipePopulation(copy.deepcopy(self.individuals), self.pop_size,
                                self.generation)

    @staticmethod
    def valid_individuals(individuals):
        return [x for x in individuals
                if getattr(x, "score", None) is not None
                and math.isfinite(float(x.score))]

    @classmethod
    def best_unique(cls, individuals, limit=None):
        # Same direction as EoH: score is maximized (usually negative cost).
        candidates = cls.valid_individuals(individuals)
        candidates.sort(key=lambda x: getattr(x, "score", float("-inf")),
                        reverse=True)
        unique, seen = [], set()
        for item in candidates:
            code = str(item)
            if code in seen:
                continue
            seen.add(code)
            unique.append(item)
            if limit is not None and len(unique) >= int(limit):
                break
        return unique

    def with_extra_reference_individuals(self, extra_individuals):
        """Return a selection-only population containing local and elite refs."""
        combined = self.best_unique(
            list(self.individuals) + list(extra_individuals),
            limit=None,
        )
        return RecipePopulation(copy.deepcopy(combined),
                                max(self.pop_size, len(combined)),
                                self.generation)

    def select_many(self, count):
        valid = self.valid_individuals(self.individuals)
        if count < 1 or count > len(valid):
            raise ValueError(f"selection_num={count} exceeds population size {len(valid)}")
        valid.sort(key=lambda x: x.score, reverse=True)
        p = np.asarray([1.0 / (i + len(valid)) for i in range(len(valid))])
        p /= p.sum()
        return list(np.random.choice(valid, size=count, replace=False, p=p))

    def survival(self, offspring):
        unique = self.best_unique(self.individuals + list(offspring),
                                  self.pop_size)
        return RecipePopulation(unique[:self.pop_size], self.pop_size,
                                self.generation + 1)

    def inherited_next_generation(self):
        """Return a valid child state when a recipe yields no feasible child."""
        return RecipePopulation(copy.deepcopy(self.individuals), self.pop_size,
                                self.generation + 1)
