"""Run MCTS-AHD-Recipe on constructive TSP."""

from pathlib import Path
import copy
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from example.tasks.mcts_recipe_runner import OpenAIEmbedding
from example.tasks.mcts_recipe_runner import write_experiment_config_snapshot
from example.tasks.tsp_construct.mcts_ahd_recipe_config import (
    MCTS_AHD_RECIPE_CONFIG,
)
from example.tasks.utils import get_info
from llm4ad.method.mcts_ahd import MAProfiler
from llm4ad.method.mcts_ahd_recipe import MCTS_AHD_Recipe
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def _section(config, name):
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"config['{name}'] must be a dict")
    return dict(value)


def _secret(config):
    value = config.get("api_key")
    if value:
        return value
    env_names = config.get("api_key_env", [])
    if isinstance(env_names, str):
        env_names = [env_names]
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    return str(path if path.is_absolute() else ROOT / path)


def main():
    config = copy.deepcopy(MCTS_AHD_RECIPE_CONFIG)
    method_config = _section(config, "method")
    llm_config = _section(config, "llm")
    embedding_config = _section(config, "embedding")
    evaluation_config = _section(config, "evaluation")

    log_dir = _resolve_path(config.get("log_dir", "logs/mcts_ahd_recipe_tsp"))
    state_dir = _resolve_path(config.get(
        "state_dir", str(Path(log_dir) / "mcts_ahd_recipe_state")))
    config["log_dir"] = log_dir
    config["state_dir"] = state_dir
    config["evaluation"]["dataset_path"] = _resolve_path(
        evaluation_config.get("dataset_path"))

    info = get_info(
        "select_next_node",
        "llm4ad.task.optimization.tsp_construct.template",
    )
    evaluation = TSPEvaluation(
        timeout_seconds=int(evaluation_config.get("timeout_seconds", 300)),
        n_instance=int(evaluation_config.get("n_instance", 50)),
        problem_size=int(evaluation_config.get("problem_size", 100)),
        dataset_path=evaluation_config.get("dataset_path"),
        seed=int(evaluation_config.get("seed", 2024)),
    )
    llm = HttpsApi(
        host=llm_config.get("host", "api.apilio.ai"),
        key=_secret(llm_config),
        model=llm_config.get("model", "gpt-4o-mini"),
        timeout=int(llm_config.get("timeout", 60)),
        debug_mode=bool(method_config.get("debug", False)),
    )

    embedding_model = None
    if embedding_config.get("enabled", False):
        embedding_model = OpenAIEmbedding(
            api_key=_secret(embedding_config),
            base_url=embedding_config.get(
                "base_url", "https://api.apilio.ai/v1"),
            model=embedding_config.get("model", "text-embedding-v4"),
            encoding_format=embedding_config.get("encoding_format", "float"),
        )

    write_experiment_config_snapshot(
        log_dir,
        experiment_config=config,
        method_name="select_next_node",
        template_module="llm4ad.task.optimization.tsp_construct.template",
        evaluation=evaluation,
        resolved_options={
            "state_dir": state_dir,
            "resume_mode": bool(config.get("resume_mode", False)),
            **method_config,
        },
    )

    method = MCTS_AHD_Recipe(
        llm=llm,
        evaluation=evaluation,
        profiler=MAProfiler(
            log_dir=log_dir,
            log_style=method_config.get("log_style", "complex"),
            create_random_path=False,
        ),
        info=info,
        recipe_configs=config.get("recipes"),
        max_sample_nums=int(method_config.get("max_sample_nums", 500)),
        init_size=int(method_config.get("init_size", 4)),
        pop_size=int(method_config.get("pop_size", 10)),
        selection_num=int(method_config.get("selection_num", 2)),
        num_samplers=int(method_config.get("num_samplers", 10)),
        num_evaluators=int(method_config.get("num_evaluators", 10)),
        multi_thread_or_process_eval=method_config.get(
            "multi_thread_or_process_eval", "thread"),
        alpha=float(method_config.get("alpha", 0.5)),
        lambda_0=float(method_config.get("lambda_0", 0.1)),
        recipe_temperature=float(
            method_config.get("recipe_temperature", 1.0)),
        recipe_batch_expansion=bool(
            method_config.get("recipe_batch_expansion", True)),
        recipes_per_node=int(method_config.get("recipes_per_node", 5)),
        expansion_workers=int(method_config.get("expansion_workers", 30)),
        embedding_model=embedding_model,
        state_dir=state_dir,
        node_batch_size=int(method_config.get("node_batch_size", 100)),
        resume_mode=bool(config.get("resume_mode", False)),
        debug_mode=bool(method_config.get("debug", False)),
    )
    method.run()


if __name__ == "__main__":
    main()
