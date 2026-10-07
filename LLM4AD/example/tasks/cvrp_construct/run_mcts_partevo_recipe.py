"""Run MCTS-PartEvo-Recipe on constructive CVRP."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.cvrp_construct.mcts_partevo_recipe_config import (
    MCTS_PARTEVO_RECIPE_CONFIG,
)
from example.tasks.mcts_partevo_recipe_runner import build_llm, run_task
from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation


def main():
    config = MCTS_PARTEVO_RECIPE_CONFIG
    evaluation_config = config["evaluation"]
    evaluation = CVRPEvaluation(
        timeout_seconds=int(evaluation_config.get("timeout_seconds", 300)),
        problem_size=int(evaluation_config.get("problem_size", 100)),
        n_instance=int(evaluation_config.get("n_instance", 50)),
        capacity=int(evaluation_config.get("capacity", 50)),
        dataset_path=evaluation_config.get("dataset_path"),
    )
    run_task(
        method_name="select_next_node",
        template_module="llm4ad.task.optimization.cvrp_construct.template",
        evaluation=evaluation,
        llm=build_llm(config),
        config=config,
    )


if __name__ == "__main__":
    main()
