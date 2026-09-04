"""Default recipe set for recipe-level MCTS experiments."""

from __future__ import annotations

import copy


# Hold input information constant where the reflection behavior permits it.
# These values match the fixed input setting used by the 47-method ablation.
_FIXED_INPUT = {
    "parent_info": True,
    "fitness": 2,
    "avg_fitness": True,
    "check_guidance": True,
    "identical_parents": True,
    "shared_parent": True,
    "population_comparison": "elite_worst",
    "best_worst": False,
}


DEFAULT_RECIPES = {
    # Best mean training score among the existing two-run 59-config results.
    "full_difference_guidance": {
        **_FIXED_INPUT,
        "comparison": True,
        "attribution": True,
        "summarization": True,
        "attribution_type": "difference",
        "summarization_type": "guidance",
        "refineevo_experience": False,
    },
    # Strong result with a different attribution/summary objective.
    "attribution_both_conditions": {
        **_FIXED_INPUT,
        "comparison": False,
        "attribution": True,
        "summarization": True,
        "attribution_type": "both",
        "summarization_type": "conditions",
        "refineevo_experience": False,
    },
    # A compact attribution-only recipe that also performed well.
    "attribution_good": {
        **_FIXED_INPUT,
        "comparison": False,
        "attribution": True,
        "summarization": False,
        "attribution_type": "good",
        "summarization_type": "guidance",  # Ignored while summarization=False.
        "refineevo_experience": False,
    },
    # Behaviorally distinct control: comparison without attribution/summary.
    "comparison_only": {
        **_FIXED_INPUT,
        "comparison": True,
        "attribution": False,
        "summarization": False,
        "attribution_type": "good",        # Ignored while attribution=False.
        "summarization_type": "guidance",  # Ignored while summarization=False.
        "refineevo_experience": False,
    },
    # RefineEvo-style node-local experience distillation and retrieval.
    "refineevo_experience": {
        "parent_info": True,
        "fitness": 2,
        "avg_fitness": False,
        "check_guidance": True,
        "identical_parents": False,
        "shared_parent": False,
        "population_comparison": None,
        "best_worst": False,
        "comparison": False,
        "attribution": False,
        "summarization": False,
        "attribution_type": "good",
        "summarization_type": "guidance",
        "refineevo_experience": True,
        "experience_top_k": 3,
        "experience_filter": "all",
        "match_operator_goal": True,
    },
}


def get_default_recipes():
    """Return an isolated copy so experiments cannot mutate global defaults."""
    return copy.deepcopy(DEFAULT_RECIPES)


def validate_recipes(recipes):
    if len(recipes) < 2:
        raise ValueError("Recipe-MCTS requires at least two recipes.")
    for recipe_id, values in recipes.items():
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValueError("Each recipe_id must be a non-empty string.")
        if values.get("refineevo_experience", False):
            if int(values.get("experience_top_k", 0)) < 1:
                raise ValueError(f"{recipe_id}: experience_top_k must be positive.")
            continue
        if not any(values.get(key, False) for key in (
                "comparison", "attribution", "summarization")):
            raise ValueError(f"{recipe_id}: enable at least one reflection behavior.")
    return recipes
