"""Shared runner for MCTS-PartEvo-Recipe experiments."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from llm4ad.method.mcts_recipe import MCTSPartEvoRecipe, get_default_recipes
from example.tasks.mcts_recipe_runner import (
    _initial_population,
    _section,
    _secret,
    OpenAIEmbedding,
    build_llm,
    run_post_training_test,
    write_experiment_config_snapshot,
)
from example.tasks.utils import get_info


def run_task(*, method_name, template_module, evaluation, llm, config):
    config = copy.deepcopy(config)
    method_config = _section(config, "mcts_partevo")
    log_dir = Path(config.get("log_dir", "logs/mcts_partevo_recipe"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parents[2] / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    recipes = copy.deepcopy(config.get("recipes") or get_default_recipes())
    info = get_info(method_name, template_module)
    info["template_module"] = template_module
    write_experiment_config_snapshot(
        log_dir,
        experiment_config=config,
        method_name=method_name,
        template_module=template_module,
        evaluation=evaluation,
        resolved_options={
            "log_dir": str(log_dir),
            "checkpoint": config.get("checkpoint"),
            **method_config,
        },
        recipes=recipes,
    )

    method = MCTSPartEvoRecipe(
        llm=llm,
        evaluation=evaluation,
        info=info,
        recipes=recipes,
        pop_size=int(method_config.get("pop_size", 16)),
        partition_number=int(method_config.get("partition_number", 4)),
        operators=tuple(method_config.get(
            "operators", ("re", "se", "cn", "lge"))),
        operators_parent_num=method_config.get("operators_parent_num"),
        operators_frequency=method_config.get("operators_frequency"),
        num_samplers=int(method_config.get("num_samplers", 16)),
        num_evaluators=int(method_config.get("num_evaluators", 16)),
        max_round_attempts=int(method_config.get("max_round_attempts", 3)),
        feature_used=tuple(method_config.get("feature_used", ("ast",))),
        use_resource_tilt=bool(method_config.get("use_resource_tilt", False)),
        bert_model_path=method_config.get("bert_model_path"),
        max_depth=int(method_config.get("max_depth", 50)),
        max_sample_count=method_config.get("max_sample_count", 10000),
        exploration_constant=float(
            method_config.get("exploration_constant", 0.2)),
        depth_balance_weight=float(
            method_config.get("depth_balance_weight", 0.4)),
        node_batch_size=int(method_config.get("node_batch_size", 100)),
        store_dir=str(log_dir),
        seed=method_config.get("seed"),
        adaptive_config=method_config.get("adaptive"),
    )
    embedding_config = _section(config, "embedding")
    if embedding_config.get("enabled", False):
        method.set_experience_embedding_model(OpenAIEmbedding(
            api_key=_secret(embedding_config),
            base_url=embedding_config.get("base_url", "https://api.apilio.ai/v1"),
            model=embedding_config.get("model", "text-embedding-v4"),
            encoding_format=embedding_config.get("encoding_format", "float"),
        ))

    checkpoint = config.get("checkpoint")
    try:
        if checkpoint:
            return method.run(checkpoint=str(checkpoint))

        init_pop_size = int(method_config.get(
            "init_pop_size", method_config.get("pop_size", 16)))
        initial, initial_token_usage = _initial_population(
            llm,
            evaluation,
            info,
            int(method_config.get("pop_size", 16)),
            selection_num=1,
            num_samplers=int(method_config.get("initial_num_samplers", 16)),
            debug=bool(method_config.get("debug", False)),
            on_evaluated=lambda function: method.record_evaluated_sample(
                function, population_node_id=1, depth=0),
            admit_sample_order=method.admit_evaluation_sample,
            init_pop_size=init_pop_size,
        )
        result = method.run(
            initial_individuals=initial,
            initial_samples_recorded=True,
            initial_token_usage=initial_token_usage,
        )
        run_post_training_test(
            config.get("task", "tsp"),
            str(log_dir),
            _section(config, "test"),
        )
        return result
    finally:
        method.close()
        llm.close()


__all__ = ["run_task", "build_llm"]
