"""Run FunSearch with optional tree-based reflection on constructive TSP."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from funsearch_config import FUNSEARCH_CONFIG
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.funsearch.profiler import FunSearchProfiler
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def _get_secret(config: dict) -> str:
    value = config.get("api_key", "")
    if value:
        return value
    env_names = config.get("api_key_env", [])
    if isinstance(env_names, str):
        env_names = [env_names]
    for env_name in env_names:
        value = os.environ.get(env_name, "")
        if value:
            return value
    return ""


def main() -> None:
    config = FUNSEARCH_CONFIG
    llm_config = config["llm"]
    eval_config = config["evaluation"]

    evaluation = TSPEvaluation(**eval_config)
    llm = HttpsApi(
        host=llm_config["host"],
        key=_get_secret(llm_config),
        model=llm_config["model"],
        timeout=int(llm_config["timeout"]),
    )

    profiler = FunSearchProfiler(
        log_dir=config["log_dir"],
        log_style=config["profiler_log_style"],
        program_db_register_interval=int(config["program_db_register_interval"]),
        checkpoint_interval=int(config.get("checkpoint_interval", 100)),
        convergence_csv_path=config.get("convergence_csv_path"),
        convergence_png_path=config.get("convergence_png_path"),
        create_random_path=False,
    )

    method = FunSearch(
        llm=llm,
        profiler=profiler,
        evaluation=evaluation,
        max_sample_nums=config["max_sample_nums"],
        num_samplers=int(config["num_samplers"]),
        num_evaluators=int(config["num_evaluators"]),
        samples_per_prompt=int(config["samples_per_prompt"]),
        resume_mode=bool(config["resume_mode"]),
        debug_mode=bool(config["debug_mode"]),
        multi_thread_or_process_eval=config["multi_thread_or_process_eval"],
        tree_reflection=config["tree_reflection"],
        checkpoint_path=config.get("checkpoint_path"),
    )
    method.run()


if __name__ == "__main__":
    main()
