"""Run Recipe-MCTS on the repository's constructive knapsack task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.knapsack_construct import KnapsackEvaluation
from example.tasks.kp_constructive.mcts_recipe_config import MCTS_RECIPE_CONFIG
from example.tasks.mcts_recipe_runner import (
    build_llm,
    common_options,
    evaluation_config,
    run_task,
)


def main():
    config = MCTS_RECIPE_CONFIG
    options = common_options("logs/mcts_recipe_kp", config)
    evaluation = KnapsackEvaluation(**evaluation_config(config))
    llm = build_llm(config)
    run_task(
        method_name="select_next_item",
        template_module="llm4ad.task.optimization.knapsack_construct.template",
        evaluation=evaluation,
        llm=llm,
        **options,
    )


if __name__ == "__main__":
    main()
