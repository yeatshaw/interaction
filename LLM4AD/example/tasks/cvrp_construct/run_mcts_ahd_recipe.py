"""Run MCTS-AHD-Recipe on the repository's constructive CVRP task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.cvrp_construct.mcts_ahd_recipe_config import (
    MCTS_AHD_RECIPE_CONFIG,
)
from example.tasks.mcts_ahd_recipe_runner import run_task
from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation


def main():
    run_task(
        MCTS_AHD_RECIPE_CONFIG,
        task="cvrp",
        method_name="select_next_node",
        template_module="llm4ad.task.optimization.cvrp_construct.template",
        evaluation_class=CVRPEvaluation,
        evaluation_options=lambda config: {
            "timeout_seconds": int(config.get("timeout_seconds", 300)),
            "problem_size": int(config.get("problem_size", 100)),
            "n_instance": int(config.get("n_instance", 50)),
            "capacity": int(config.get("capacity", 50)),
            "dataset_path": config.get("dataset_path"),
        },
    )


if __name__ == "__main__":
    main()
