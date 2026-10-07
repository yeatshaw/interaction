from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
        functions_per_prompt: Number of previous programs to include in prompts.
        num_islands: Number of islands to maintain as a diversity mechanism.
        reset_period: How often (in seconds) the weakest islands should be reset.
        cluster_sampling_temperature_init: Initial temperature for softmax sampling
            of clusters within an island.
        cluster_sampling_temperature_period: Period of linear decay of the cluster
            sampling temperature.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class TreeReflectionConfig:
    """Optional tree-based reflection configuration for FunSearch."""

    enabled: bool = False
    reference_mode: str = 'best'
    adaptive_recipe_selection: bool = True
    recipes: tuple[str, ...] = ('comparison', 'attribution', 'summarization')
    prior_precision: float = 1.0
    posterior_scale: float = 0.5
    reward_scale: float = 1.0
    max_parent_context: int = 2
    experience_top_k: int = 3
    experience_path: str | None = None
    embedding: dict[str, Any] = dataclasses.field(default_factory=dict)
    tree_path: str | None = None
    stats_path: str | None = None
    tree_persist_interval: int = 100

    @classmethod
    def from_value(cls, value: Any) -> 'TreeReflectionConfig':
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError('tree_reflection must be a TreeReflectionConfig or dict.')
        payload = dict(value)
        if 'recipes' in payload and payload['recipes'] is not None:
            payload['recipes'] = tuple(payload['recipes'])
        return cls(**payload)

    def __post_init__(self) -> None:
        if self.reference_mode not in {'best', 'random', 'lineage_parent'}:
            raise ValueError(
                'reference_mode must be one of: best, random, lineage_parent.')
        if not self.recipes:
            raise ValueError('tree_reflection requires at least one recipe.')
        if self.prior_precision <= 0:
            raise ValueError('prior_precision must be positive.')
        if self.posterior_scale < 0:
            raise ValueError('posterior_scale must be non-negative.')
        if self.max_parent_context < 0:
            raise ValueError('max_parent_context must be non-negative.')
        if self.experience_top_k < 1:
            raise ValueError('experience_top_k must be positive.')
        if self.tree_persist_interval < 1:
            raise ValueError('tree_persist_interval must be positive.')
