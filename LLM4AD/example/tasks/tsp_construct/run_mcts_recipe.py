"""Run recipe-MCTS for the constructive TSP task."""

from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from example.tasks.mcts_recipe_runner import common_options, run_task


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recipe-MCTS on TSP.")
    parser.add_argument(
        "--train-data",
        default=os.environ.get("LLM4AD_TSP_TRAIN_DATA"),
        help=("Path to the pickle training dataset. Falls back to "
              "LLM4AD_TSP_TRAIN_DATA; random data is used when omitted."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    options = common_options("logs/mcts_recipe_tsp")
    evaluation = TSPEvaluation(
        timeout_seconds=int(os.environ.get("LLM4AD_TIMEOUT", "300")),
        n_instance=int(os.environ.get("LLM4AD_N_INSTANCE", "50")),
        problem_size=int(os.environ.get("LLM4AD_PROBLEM_SIZE", "100")),
        dataset_path=args.train_data,
        seed=int(os.environ.get("LLM4AD_DATA_SEED", "2024")),
    )
    llm = HttpsApi(
        host=os.environ.get("LLM4AD_API_HOST", "api.apilio.ai"),
        key=os.environ.get("LLM4AD_API_KEY", ""),
        model=os.environ.get("LLM4AD_API_MODEL", "gpt-4o-mini"),
        timeout=int(os.environ.get("LLM4AD_API_TIMEOUT", "60")),
    )
    run_task(
        method_name="select_next_node",
        template_module="llm4ad.task.optimization.tsp_construct.template",
        evaluation=evaluation,
        llm=llm,
        **options,
    )


if __name__ == "__main__":
    main()
