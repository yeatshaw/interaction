"""Run Recipe-MCTS on the repository's constructive 1D bin-packing task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.bp_1d_construct import BP1DEvaluation
from example.tasks.bp_1d_construct.mcts_recipe_config import MCTS_RECIPE_CONFIG
from example.tasks.mcts_recipe_runner import (
    build_llm,
    common_options,
    evaluation_config,
    run_task,
)


def main():
    config = MCTS_RECIPE_CONFIG
    options = common_options("logs/mcts_recipe_bp_1d", config)
    evaluation = BP1DEvaluation(**evaluation_config(config))
    llm = build_llm(config)
    run_task(
        method_name="determine_next_assignment",
        template_module="llm4ad.task.optimization.bp_1d_construct.template",
        evaluation=evaluation,
        llm=llm,
        task_type="bp_1d",
        **options,
    )


if __name__ == "__main__":
    main()
