"""Shared command-line runner for task-specific Recipe-MCTS experiments."""

from __future__ import annotations

import os
import time
import concurrent.futures
import math
import json
import urllib.request
from pathlib import Path

import numpy as np

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.eoh.prompt import EoHPrompt
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.method.mcts_recipe import (
    EoHRecipeExpander,
    MCTSRecipe,
    RefineEvoRecipeExpander,
    RefineEvoExperienceManager,
    get_default_recipes,
)


class OpenAIEmbedding:
    """Small OpenAI-compatible embedding adapter used by RefineEvo."""

    def __init__(self, api_key, base_url, model, encoding_format="float"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.encoding_format = encoding_format

    def encode(self, texts):
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({
                "model": self.model,
                "input": inputs,
                "encoding_format": self.encoding_format,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise RuntimeError(f"Invalid embedding response: {str(payload)[:500]}")
        ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
        vectors = np.asarray([item["embedding"] for item in ordered], dtype=float)
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f"Embedding service returned {len(vectors)} vectors for {len(inputs)} inputs")
        return vectors[0] if single else vectors


def _int_env(name, default):
    return int(os.environ.get(name, str(default)))


def _initial_population(llm, evaluation, info, pop_size, selection_num=2,
                        num_samplers=1, debug=False):
    """Generate and evaluate the root population with the EoH i1 prompt."""
    sampler = EoHSampler(llm, info["template_program"])
    evaluator = SecureEvaluator(evaluation, debug_mode=debug)
    individuals = []
    prompt = EoHPrompt.get_prompt_i1(info)
    max_attempts = pop_size * 2

    def sample_one():
        sample_start = time.time()
        thought, function = sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or function is None:
            return None
        program = TextFunctionProgramConverter.function_to_program(
            function, info["template_program"])
        if program is None:
            return None
        score, eval_time = evaluator.evaluate_program_record_time(program)
        if score is None or not isinstance(score, (int, float)) \
                or not math.isfinite(float(score)):
            if debug and score is not None:
                print(f"DEBUG: rejected non-finite initialization score: {score}")
            return None
        function.algorithm = thought
        function.score = score
        function.evaluate_time = eval_time
        function.sample_time = sample_time
        function.operator = "i1"
        return function

    attempts = 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, num_samplers)) as executor:
        while len(individuals) < pop_size and attempts < max_attempts:
            batch_size = min(max(1, num_samplers), max_attempts - attempts)
            attempts += batch_size
            for function in executor.map(lambda _: sample_one(), range(batch_size)):
                if function is not None and len(individuals) < pop_size:
                    individuals.append(function)
                    print(f"Initialized root individual {len(individuals)}/{pop_size}, "
                          f"score={function.score}")
    if len(individuals) < selection_num:
        raise RuntimeError(
            f"Initialization produced only {len(individuals)} feasible algorithms "
            f"after {max_attempts} samples; at least {selection_num} are required")
    if len(individuals) < pop_size:
        print(f"Initialization stopped after {max_attempts} samples with "
              f"{len(individuals)}/{pop_size} feasible algorithms")
    return individuals


def run_task(*, method_name, template_module, evaluation, llm,
             store_dir, pop_size=10, selection_num=2, max_depth=50,
             num_samplers=10, num_evaluators=10, debug=False):
    from example.tasks.utils import get_info

    info = get_info(method_name, template_module)
    checkpoint = os.environ.get("LLM4AD_CHECKPOINT")
    if checkpoint:
        store_dir = Path(checkpoint).resolve().parent
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recipe-MCTS output directory: {store_dir}", flush=True)
    initial = (None if checkpoint else
               _initial_population(llm, evaluation, info, pop_size,
                                   selection_num, num_samplers, debug))
    recipes = get_default_recipes()
    embedding_host = os.environ.get(
        "LLM4AD_EMBEDDING_BASE_URL",
        f"https://{os.environ.get('LLM4AD_API_HOST', 'api.apilio.ai')}/v1")
    embedding = OpenAIEmbedding(
        api_key=os.environ.get("LLM4AD_EMBEDDING_API_KEY",
                               os.environ.get("LLM4AD_API_KEY", "")),
        base_url=embedding_host,
        model=os.environ.get(
            "LLM4AD_EMBEDDING_MODEL", "text-embedding-3-small"),
        encoding_format=os.environ.get(
            "LLM4AD_EMBEDDING_ENCODING_FORMAT", "float"),
    )
    experience_manager = RefineEvoExperienceManager(
        reflector_llm=llm,
        embedding_model=embedding,
        top_k=int(recipes["refineevo_experience"]["experience_top_k"]),
    )
    expanders = {}
    for recipe_id, recipe in recipes.items():
        cls = (RefineEvoRecipeExpander
               if recipe["refineevo_experience"] else EoHRecipeExpander)
        kwargs = {}
        if recipe["refineevo_experience"]:
            kwargs.update(
                retrieve_experiences=experience_manager.retrieve,
                distill_experience=experience_manager.distill,
            )
        expanders[recipe_id] = cls(
            llm=llm,
            evaluation=evaluation,
            info=info,
            template_program=info["template_program"],
            num_samplers=num_samplers,
            num_evaluators=num_evaluators,
            debug_mode=debug,
            **kwargs,
        )
    method = MCTSRecipe(
        recipes=recipes,
        pop_size=pop_size,
        selection_num=selection_num,
        max_depth=max_depth,
        exploration_constant=float(os.environ.get("LLM4AD_EXPLORATION", "0.1")),
        node_batch_size=_int_env("LLM4AD_NODE_BATCH_SIZE", 10),
        store_dir=str(store_dir),
        expand_fn=expanders,
        seed=(int(os.environ["LLM4AD_SEED"])
              if "LLM4AD_SEED" in os.environ else None),
    )
    try:
        return method.run(initial_individuals=initial, checkpoint=checkpoint)
    finally:
        for expander in expanders.values():
            expander.close(close_llm=False)
        llm.close()


def common_options(default_log):
    configured_log = os.environ.get("LLM4AD_LOG_DIR")
    repository_root = Path(__file__).resolve().parents[2]
    store_dir = (Path(configured_log) if configured_log
                 else repository_root / default_log)
    return {
        "store_dir": store_dir,
        "pop_size": _int_env("LLM4AD_POP_SIZE", 10),
        "selection_num": _int_env("LLM4AD_SELECTION_NUM", 2),
        "max_depth": _int_env("LLM4AD_MAX_DEPTH", 50),
        "num_samplers": _int_env("LLM4AD_NUM_SAMPLERS", 10),
        # Keep generation and evaluation parallel by default.  Set this to a
        # smaller value on machines where each safe evaluation starts a child
        # process or the task itself uses many CPU threads.
        "num_evaluators": _int_env("LLM4AD_NUM_EVALUATORS", 10),
        "debug": os.environ.get("LLM4AD_DEBUG", "0") == "1",
    }
