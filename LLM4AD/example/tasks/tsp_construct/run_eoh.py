from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.utils import get_info
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def main():
    log_dir = Path(os.environ.get("LLM4AD_LOG_DIR", "logs/eoh_tsp"))
    evaluation = TSPEvaluation(
        timeout_seconds=int(os.environ.get("LLM4AD_TIMEOUT", "300")),
        n_instance=int(os.environ.get("LLM4AD_N_INSTANCE", "50")),
        problem_size=int(os.environ.get("LLM4AD_PROBLEM_SIZE", "100")),
        dataset_path=os.environ.get("LLM4AD_TSP_TRAIN_DATA"),
        seed=int(os.environ.get("LLM4AD_DATA_SEED", "2024")),
    )
    llm = HttpsApi(
        host=os.environ.get("LLM4AD_API_HOST", "api.apilio.ai"),
        key=os.environ.get("LLM4AD_API_KEY", ""),
        model=os.environ.get("LLM4AD_API_MODEL", "gpt-4o-mini"),
        timeout=int(os.environ.get("LLM4AD_API_TIMEOUT", "60")),
    )
    info = get_info(
        "select_next_node",
        "llm4ad.task.optimization.tsp_construct.template",
    )
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir=str(log_dir), log_style="complex"),
        evaluation=evaluation,
        max_sample_nums=int(os.environ.get("LLM4AD_MAX_SAMPLES", "500")),
        max_generations=int(os.environ.get("LLM4AD_MAX_GENERATIONS", "50")),
        pop_size=int(os.environ.get("LLM4AD_POP_SIZE", "10")),
        selection_num=int(os.environ.get("LLM4AD_SELECTION_NUM", "2")),
        num_samplers=int(os.environ.get("LLM4AD_NUM_SAMPLERS", "10")),
        num_evaluators=int(os.environ.get("LLM4AD_NUM_EVALUATORS", "10")),
        lineage_log_path=os.environ.get("LLM4AD_EOH_LINEAGE", "eoh_lineage.json"),
        info=info,
        debug_mode=os.environ.get("LLM4AD_DEBUG", "0") == "1",
    )
    method.run()


if __name__ == "__main__":
    main()
