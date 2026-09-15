"""Shared command-line runner for task-specific Recipe-MCTS experiments."""

from __future__ import annotations

import os
import time
import concurrent.futures
import math
import json
import threading
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.method.mcts_recipe import (
    EoHRecipeExpander,
    MCTSRecipe,
    NoReflectionEoHRecipeExpander,
    RefineEvoRecipeExpander,
    RefineEvoExperienceManager,
    get_default_recipes,
)


REFINEEVO_SYSTEM_GENERATOR_TEMPLATE = """You are a world-class expert in optimization algorithms and heuristic design with deep expertise in operations research, combinatorial optimization, and metaheuristic methods.

## YOUR ROLE
Design novel, efficient, and effective heuristic algorithms to solve complex optimization problems.

## PROBLEM CONTEXT
**Function Name:** {func_name}
**Problem Description:** {problem_desc}
**Detailed Specification:**
{func_desc}

## ALGORITHM DESIGN PRINCIPLES
1. **Novelty**: Create algorithms with unique characteristics that differ from existing approaches
2. **Efficiency**: Ensure computational efficiency suitable for the problem scale
3. **Robustness**: Design algorithms that perform well across diverse problem instances

## OUTPUT FORMAT REQUIREMENTS
You must provide your response in exactly two parts:

**Part 1: Algorithm Description**
- Write a concise one-sentence description of your algorithm
- Enclose the description in curly braces: {{your algorithm description here}}
- Focus on the core innovation and key mechanism

**Part 2: Python Implementation**
- Provide complete, runnable Python code
- Enclose code in: ```python ... ```
- Follow the exact function signature specified
- Include necessary imports within the function
- Use clear variable names and efficient data structures
- Ensure code is production-ready without bugs

## IMPORTANT CONSTRAINTS
- Do NOT provide any additional explanations, commentary, or discussion
- Do NOT include example usage or test cases
- Output ONLY the algorithm description in braces and the Python code block
- Ensure the code is syntactically correct and logically sound"""


REFINEEVO_USER_INIT_TEMPLATE = """## TASK: Generate Initial Algorithm

You are tasked with designing a novel heuristic algorithm from scratch. This is the initial design phase where you will create the foundation for the evolutionary algorithm development process.

### REFERENCE IMPLEMENTATION
Below is a baseline implementation to illustrate the expected code structure and format:

{seed_func}

### ADDITIONAL DOMAIN KNOWLEDGE
{external_knowledge}
"""


REFINEEVO_PROBLEM_ALIASES = {
    "tsp_construct": "tsp_constructive",
    "cvrp_construct": "cvrp_constructive",
    "vrptw_construct": "vrptw_constructive",
    "kp_construct": "kp_constructive",
}


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


def _read_text_if_exists(path):
    try:
        path = Path(path)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def _refineevo_prompt_root():
    configured = os.environ.get("LLM4AD_REFINEEVO_ROOT")
    if configured:
        return Path(configured) / "prompts"
    configured = os.environ.get("LLM4AD_REFINEEVO_PROMPT_DIR")
    if configured:
        return Path(configured)
    repository_root = Path(__file__).resolve().parents[2]
    workspace_root = repository_root.parent.parent
    return workspace_root / "RefineEvo" / "prompts"


def _refineevo_problem_name(template_module):
    configured = os.environ.get("LLM4AD_REFINEEVO_PROBLEM")
    if configured:
        return configured.strip()
    task_name = template_module.split(".")[-2]
    return REFINEEVO_PROBLEM_ALIASES.get(task_name, task_name)


def _refineevo_initial_prompt(info, template_module):
    prompt_root = _refineevo_prompt_root()
    common_dir = prompt_root / "common"
    problem_dir = prompt_root / _refineevo_problem_name(template_module)

    system_template = (_read_text_if_exists(common_dir / "system_generator.txt")
                       or REFINEEVO_SYSTEM_GENERATOR_TEMPLATE)
    user_template = (_read_text_if_exists(common_dir / "user_init.txt")
                     or REFINEEVO_USER_INIT_TEMPLATE)

    func_desc = (_read_text_if_exists(problem_dir / "func_desc.txt")
                 or info.get("method_args")
                 or info.get("task_description", ""))
    seed_func = (_read_text_if_exists(problem_dir / "seed_func.txt")
                 or info.get("seed_func")
                 or info["template_program"])
    external_knowledge = (
        _read_text_if_exists(problem_dir / "external_knowledge.txt")
        or info.get("external_knowledge")
        or "No additional domain knowledge provided."
    )

    system = system_template.format(
        func_name=info["method_name"],
        problem_desc=info["task_description"],
        func_desc=func_desc,
    )
    user = user_template.format(
        seed_func=seed_func,
        external_knowledge=external_knowledge,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _initial_population(llm, evaluation, info, pop_size, selection_num=2,
                        num_samplers=1, debug=False, on_evaluated=None,
                        reserve_sample_order=None):
    """Generate and evaluate the root population with the RefineEvo init prompt."""
    sampler = EoHSampler(llm, info["template_program"])
    evaluator = SecureEvaluator(evaluation, debug_mode=debug)
    individuals = []
    prompt = _refineevo_initial_prompt(info, info["template_module"])
    max_attempts = pop_size * 2
    token_usage_records = []
    token_usage_lock = threading.RLock()

    def capture_token_usage():
        capture = getattr(llm, "capture_token_usage", None)
        if callable(capture):
            return capture()
        return nullcontext({})

    def compact_token_usage(usage):
        compact = getattr(llm, "compact_token_usage", None)
        if callable(compact):
            return compact(usage)
        return usage or None

    def record_token_usage(usage):
        usage = compact_token_usage(usage)
        if usage:
            with token_usage_lock:
                token_usage_records.append(usage)
        return usage

    def merge_token_usage():
        with token_usage_lock:
            records = list(token_usage_records)
        merge = getattr(llm, "merge_token_usage", None)
        if callable(merge):
            usage = merge(*records)
            return compact_token_usage(usage) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        return {
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in records),
            "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in records),
        }

    def sample_one():
        sample_order = (reserve_sample_order()
                        if reserve_sample_order is not None else None)
        sample_start = time.time()
        evolution_usage = None
        token_usage = None
        try:
            with capture_token_usage() as evolution_usage:
                thought, function = sampler.get_thought_and_function(prompt)
        finally:
            token_usage = record_token_usage(evolution_usage)
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
        function._recipe_token_usage = token_usage
        function.operator = "i1"
        function._recipe_sample_order = sample_order
        if on_evaluated is not None:
            on_evaluated(function)
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
    return individuals, merge_token_usage()


def run_task(*, method_name, template_module, evaluation, llm,
             store_dir, pop_size=10, selection_num=2, max_depth=50,
             num_samplers=10, num_evaluators=10, debug=False):
    from example.tasks.utils import get_info

    info = get_info(method_name, template_module)
    info["template_module"] = template_module
    checkpoint = os.environ.get("LLM4AD_CHECKPOINT")
    if checkpoint:
        store_dir = Path(checkpoint).resolve().parent
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recipe-MCTS output directory: {store_dir}", flush=True)
    recipes = get_default_recipes()
    print(
        f"Recipe-MCTS recipes ({len(recipes)}): {', '.join(recipes)}",
        flush=True,
    )
    mode = os.environ.get(
        "LLM4AD_MCTS_RECIPE_MODE",
        "no_reflection" if os.environ.get("LLM4AD_NO_REFLECTION", "0") == "1"
        else "reflection",
    ).strip().lower()
    if mode not in {"reflection", "no_reflection"}:
        raise ValueError(
            "LLM4AD_MCTS_RECIPE_MODE must be 'reflection' or 'no_reflection'")
    print(f"Recipe-MCTS mode: {mode}", flush=True)

    experience_manager = None
    if mode == "reflection":
        embedding_host = os.environ.get(
            "LLM4AD_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1")
        embedding = OpenAIEmbedding(
            api_key=os.environ.get(
                "LLM4AD_EMBEDDING_API_KEY",
                os.environ.get("OPENAI_API_KEY8",
                               os.environ.get("LLM4AD_API_KEY", ""))),
            base_url=embedding_host,
            model=os.environ.get(
                "LLM4AD_EMBEDDING_MODEL", "text-embedding-v4"),
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
        if mode == "no_reflection":
            cls = NoReflectionEoHRecipeExpander
        else:
            cls = (RefineEvoRecipeExpander
                   if recipe["refineevo_experience"] else EoHRecipeExpander)
        kwargs = {}
        if mode == "reflection" and recipe["refineevo_experience"]:
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
        depth_balance_weight=float(os.environ.get(
            "LLM4AD_DEPTH_BALANCE_WEIGHT",
            os.environ.get("LLM4AD_DEPTH_BIAS", "0.2"))),
        node_batch_size=_int_env("LLM4AD_NODE_BATCH_SIZE", 10),
        store_dir=str(store_dir),
        expand_fn=expanders,
        seed=(int(os.environ["LLM4AD_SEED"])
              if "LLM4AD_SEED" in os.environ else None),
    )
    try:
        if checkpoint:
            initial = None
        else:
            # Construct the recorder before sampling so every evaluated root
            # candidate can update sample_best.json immediately.
            initial, initial_token_usage = _initial_population(
                llm, evaluation, info, pop_size, selection_num,
                num_samplers, debug,
                on_evaluated=lambda function: method.record_evaluated_sample(
                    function, population_node_id=1, depth=0),
                reserve_sample_order=method.reserve_sample_order,
            )
        return method.run(
            initial_individuals=initial,
            checkpoint=checkpoint,
            # Root samples were already recorded by the callback above.
            initial_samples_recorded=(checkpoint is None),
            initial_token_usage=(None if checkpoint else initial_token_usage),
        )
    finally:
        for expander in expanders.values():
            expander.close(close_llm=False)
        llm.close()


def common_options(default_log):
    configured_log = os.environ.get("LLM4AD_LOG_DIR")
    repository_root = Path(__file__).resolve().parents[2]
    store_dir = (Path(configured_log) if configured_log
                 else repository_root / default_log)
    pop_size = _int_env("LLM4AD_POP_SIZE", 10)
    return {
        "store_dir": store_dir,
        "pop_size": pop_size,
        "selection_num": _int_env("LLM4AD_SELECTION_NUM", 2),
        "max_depth": _int_env("LLM4AD_MAX_DEPTH", 50),
        # Match EoH-style generation by default: one concurrent sampler and
        # evaluator pipeline per population slot. Both values remain
        # independently configurable for resource-constrained machines.
        "num_samplers": _int_env("LLM4AD_NUM_SAMPLERS", pop_size),
        "num_evaluators": _int_env("LLM4AD_NUM_EVALUATORS", pop_size),
        "debug": os.environ.get("LLM4AD_DEBUG", "0") == "1",
    }
