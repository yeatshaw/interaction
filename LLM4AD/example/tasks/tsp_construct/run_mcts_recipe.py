"""Run recipe-MCTS for the constructive TSP task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.tsp_construct import (
    RefineEVOTSPEvaluation,
    TSPEvaluation,
)
from example.tasks.mcts_recipe_runner import (
    build_llm,
    common_options,
    evaluation_config,
    run_task,
)
from example.tasks.tsp_construct.mcts_recipe_config import MCTS_RECIPE_CONFIG


def main():
    config = MCTS_RECIPE_CONFIG
    options = common_options("logs/mcts_recipe_tsp", config)
    eval_options = evaluation_config(config)
    evaluation_mode = str(eval_options.pop("mode", "refineevo")).lower()
    evaluation_cls = (
        TSPEvaluation if evaluation_mode in {"original", "llm4ad"}
        else RefineEVOTSPEvaluation
    )
    print(f"TSP evaluation mode: "
          f"{'refineevo' if evaluation_cls is RefineEVOTSPEvaluation else 'original'}",
          flush=True)
    evaluation = evaluation_cls(**eval_options)
    llm = build_llm(config)
    run_task(
        method_name="select_next_node",
        template_module="llm4ad.task.optimization.tsp_construct.template",
        evaluation=evaluation,
        llm=llm,
        **options,
    )


if __name__ == "__main__":
    main()
