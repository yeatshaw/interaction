"""Run MCTS-AHD with recipe-conditioned EoH reflection on TSP."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.utils import get_info
from llm4ad.method.mcts_ahd_recipe import MCTS_AHD_Recipe
from llm4ad.method.mcts_ahd import MAProfiler
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def main():
    info = get_info(
        "select_next_node",
        "llm4ad.task.optimization.tsp_construct.template",
    )
    evaluation = TSPEvaluation(
        timeout_seconds=int(os.environ.get("LLM4AD_TIMEOUT", "300")),
        n_instance=int(os.environ.get("LLM4AD_N_INSTANCE", "50")),
        problem_size=int(os.environ.get("LLM4AD_PROBLEM_SIZE", "100")),
        dataset_path=os.environ.get("LLM4AD_TSP_TRAIN_DATA"),
    )
    llm = HttpsApi(
        host=os.environ.get("LLM4AD_API_HOST", "api.apilio.ai"),
        key=os.environ.get("LLM4AD_API_KEY", ""),
        model=os.environ.get("LLM4AD_API_MODEL", "gpt-4o-mini"),
        timeout=int(os.environ.get("LLM4AD_API_TIMEOUT", "60")),
    )
    method = MCTS_AHD_Recipe(
        llm=llm,
        evaluation=evaluation,
        profiler=MAProfiler(
            log_dir=os.environ.get("LLM4AD_LOG_DIR", "logs/mcts_ahd_recipe_tsp"),
            log_style="complex",
        ),
        info=info,
        max_sample_nums=int(os.environ.get("LLM4AD_MAX_SAMPLES", "500")),
        init_size=int(os.environ.get("LLM4AD_INIT_SIZE", "4")),
        pop_size=int(os.environ.get("LLM4AD_POP_SIZE", "10")),
        selection_num=int(os.environ.get("LLM4AD_SELECTION_NUM", "2")),
        num_samplers=int(os.environ.get("LLM4AD_NUM_SAMPLERS", "10")),
        num_evaluators=int(os.environ.get("LLM4AD_NUM_EVALUATORS", "10")),
        recipe_temperature=float(os.environ.get("LLM4AD_RECIPE_TEMPERATURE", "1.0")),
        debug_mode=os.environ.get("LLM4AD_DEBUG", "0") == "1",
    )
    method.run()


if __name__ == "__main__":
    main()
