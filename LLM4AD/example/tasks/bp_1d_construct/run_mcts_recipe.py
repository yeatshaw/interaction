"""Run Recipe-MCTS on the repository's constructive 1D bin-packing task."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.bp_1d_construct import BP1DEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from example.tasks.mcts_recipe_runner import common_options, run_task


def main():
    options = common_options("logs/mcts_recipe_bp_1d")
    evaluation = BP1DEvaluation(
        timeout_seconds=int(os.environ.get("LLM4AD_TIMEOUT", "300")),
        n_instance=int(os.environ.get("LLM4AD_N_INSTANCE", "8")),
        n_items=int(os.environ.get("LLM4AD_N_ITEMS", "500")),
        n_bins=int(os.environ.get("LLM4AD_N_BINS", "500")),
        bin_capacity=int(os.environ.get("LLM4AD_BIN_CAPACITY", "100")),
    )
    llm = HttpsApi(
        host=os.environ.get("LLM4AD_API_HOST", "api.apilio.ai"),
        key=os.environ.get("LLM4AD_API_KEY", ""),
        model=os.environ.get("LLM4AD_API_MODEL", "gpt-4o-mini"),
        timeout=int(os.environ.get("LLM4AD_API_TIMEOUT", "60")),
    )
    run_task(
        method_name="determine_next_assignment",
        template_module="llm4ad.task.optimization.bp_1d_construct.template",
        evaluation=evaluation,
        llm=llm,
        **options,
    )


if __name__ == "__main__":
    main()
