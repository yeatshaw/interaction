"""Shared runner for single-path EoH with per-offspring recipe selection."""

from __future__ import annotations

import os
from pathlib import Path

from example.tasks.mcts_recipe_runner import OpenAIEmbedding
from example.tasks.utils import get_info
from llm4ad.method.adaptive_eoh import AdaptiveRecipeEoH
from llm4ad.method.eoh import EoHProfiler
from llm4ad.method.mcts_recipe import (
    RefineEvoExperienceManager,
    get_default_recipes,
)


def _int_env(name, default):
    return int(os.environ.get(name, str(default)))


def run_task(*, method_name, template_module, evaluation, llm, log_dir):
    info = get_info(method_name, template_module)
    log_dir = Path(os.environ.get("LLM4AD_LOG_DIR", log_dir)).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    profiler = EoHProfiler(
        log_dir=str(log_dir), log_style="complex", create_random_path=False)
    recipes = get_default_recipes()
    embedding = OpenAIEmbedding(
        api_key=os.environ.get(
            "LLM4AD_EMBEDDING_API_KEY",
            os.environ.get("OPENAI_API_KEY8",
                           os.environ.get("LLM4AD_API_KEY", ""))),
        base_url=os.environ.get(
            "LLM4AD_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.environ.get("LLM4AD_EMBEDDING_MODEL", "text-embedding-v4"),
        encoding_format=os.environ.get(
            "LLM4AD_EMBEDDING_ENCODING_FORMAT", "float"),
    )
    experience_manager = RefineEvoExperienceManager(
        reflector_llm=llm,
        embedding_model=embedding,
        top_k=int(recipes["refineevo_experience"]["experience_top_k"]),
    )

    method = AdaptiveRecipeEoH(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        info=info,
        recipe_configs=recipes,
        recipe_temperature=float(os.environ.get(
            "LLM4AD_RECIPE_TEMPERATURE", "1.0")),
        refineevo_manager=experience_manager,
        recipe_stats_path=str(log_dir / "adaptive_recipe_stats.json"),
        max_sample_nums=_int_env("LLM4AD_MAX_SAMPLES", 500),
        max_generations=_int_env("LLM4AD_MAX_GENERATIONS", 500),
        pop_size=_int_env("LLM4AD_POP_SIZE", 10),
        selection_num=_int_env("LLM4AD_SELECTION_NUM", 2),
        num_samplers=_int_env("LLM4AD_NUM_SAMPLERS", 10),
        num_evaluators=_int_env("LLM4AD_NUM_EVALUATORS", 10),
        use_e2_operator=os.environ.get("LLM4AD_USE_E2", "1") == "1",
        use_m1_operator=os.environ.get("LLM4AD_USE_M1", "1") == "1",
        use_m2_operator=os.environ.get("LLM4AD_USE_M2", "1") == "1",
        use_long_term_reflection=False,
        lineage_log_path=str(log_dir / "eoh_lineage.json"),
        debug_mode=os.environ.get("LLM4AD_DEBUG", "0") == "1",
    )
    print(f"Adaptive Recipe EoH output directory: {log_dir}", flush=True)
    return method.run()
