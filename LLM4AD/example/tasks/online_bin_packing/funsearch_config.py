from __future__ import annotations

from llm4ad.method.funsearch import TreeReflectionConfig


# Set enabled=True to activate tree-based reflection for this task.
TREE_REFLECTION_CONFIG = TreeReflectionConfig(
    enabled=False,
    reference_mode='best',
    adaptive_recipe_selection=True,
    recipes=('comparison', 'attribution', 'summarization'),
    prior_precision=1.0,
    posterior_scale=0.5,
    reward_scale=1.0,
    max_parent_context=2,
    tree_path=None,
    stats_path=None,
)
