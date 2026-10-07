"""Run MCTS-AHD-Recipe on the repository's constructive knapsack task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.kp_constructive.mcts_ahd_recipe_config import (
    MCTS_AHD_RECIPE_CONFIG,
)
from example.tasks.mcts_ahd_recipe_runner import run_task
from llm4ad.task.optimization.knapsack_construct import KnapsackEvaluation


def main():
    run_task(
        MCTS_AHD_RECIPE_CONFIG,
        task="kp",
        method_name="select_next_item",
        template_module="llm4ad.task.optimization.knapsack_construct.template",
        evaluation_class=KnapsackEvaluation,
        evaluation_options=lambda config: {
            "timeout_seconds": int(config.get("timeout_seconds", 300)),
            "n_instance": int(config.get("n_instance", 50)),
            "n_items": int(config.get("n_items", 100)),
            "knapsack_capacity": int(config.get("knapsack_capacity", 100)),
            "dataset_path": config.get("dataset_path"),
        },
    )


if __name__ == "__main__":
    main()
