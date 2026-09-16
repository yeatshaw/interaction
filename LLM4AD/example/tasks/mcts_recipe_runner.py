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
from llm4ad.method.eoh.prompt import EoHPrompt
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.method.mcts_recipe import (
    EoHRecipeExpander,
    MCTSRecipe,
    NoReflectionEoHRecipeExpander,
    RefineEvoRecipeExpander,
    RefineEvoExperienceManager,
    get_default_recipes,
)
from llm4ad.method.mcts_recipe.budget import SampleBudgetExhausted
from llm4ad.method.mcts_recipe.refineevo_init import (
    initial_messages,
    sample_initial_algorithm,
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


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _section(config, name):
    value = (config or {}).get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"config['{name}'] must be a dict")
    return dict(value)


def _secret(config, key="api_key", env_key="api_key_env", default=""):
    value = config.get(key)
    if value:
        return value
    env_names = config.get(env_key)
    if isinstance(env_names, str):
        env_names = [env_names]
    for env_name in env_names or []:
        value = os.environ.get(env_name)
        if value:
            return value
    return default


def build_llm(config):
    """Build the configured OpenAI-compatible HTTPS LLM client."""
    from llm4ad.tools.llm.llm_api_https import HttpsApi

    llm_config = _section(config, "llm")
    return HttpsApi(
        host=llm_config.get("host", "api.apilio.ai"),
        key=_secret(llm_config, default=""),
        model=llm_config.get("model", "gpt-4o-mini"),
        timeout=int(llm_config.get("timeout", 60)),
    )


def evaluation_config(config):
    return _section(config, "evaluation")


def _operators(config):
    configured = (config or {}).get("operators", ("e1", "e2", "m1", "m2"))
    if isinstance(configured, str):
        return tuple(x.strip() for x in configured.split(",") if x.strip())
    return tuple(configured)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _redact_secrets(value, key_name=""):
    sensitive_keys = {"api_key", "key", "token", "password", "secret"}
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lower_key = str(key).lower()
            if lower_key in sensitive_keys and item:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_secrets(item, lower_key)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item, key_name) for item in value]
    return value


def _next_experiment_config_path(store_dir):
    base_path = Path(store_dir) / "experiment_config.json"
    if not base_path.exists():
        return base_path
    index = 1
    while True:
        path = Path(store_dir) / f"experiment_config_{index:06d}.json"
        if not path.exists():
            return path
        index += 1


def write_experiment_config_snapshot(
        store_dir, *, experiment_config=None, method_name=None,
        template_module=None, evaluation=None, resolved_options=None,
        recipes=None):
    """Write one immutable JSON config snapshot into the experiment folder."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    evaluation_class = None
    if evaluation is not None:
        evaluation_class = (
            f"{evaluation.__class__.__module__}.{evaluation.__class__.__name__}")
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method_name": method_name,
        "template_module": template_module,
        "evaluation_class": evaluation_class,
        "config": experiment_config or {},
        "resolved_options": resolved_options or {},
        "recipes": recipes or {},
    }
    payload = _redact_secrets(_json_safe(payload))
    path = _next_experiment_config_path(store_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _refineevo_initial_prompt(info, template_module):
    # The initialization assets are local to LLM4AD; no runtime dependency on
    # the separate RefineEvo checkout is allowed.
    return initial_messages(info["method_name"], template_module)


def _token_usage_helpers(llm):
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
            usage = compact_token_usage(usage) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        else:
            usage = {
                "prompt_tokens": sum(
                    int(item.get("prompt_tokens", 0)) for item in records),
                "completion_tokens": sum(
                    int(item.get("completion_tokens", 0)) for item in records),
            }
        usage["total_tokens"] = (
            int(usage.get("prompt_tokens", 0))
            + int(usage.get("completion_tokens", 0)))
        return usage

    return capture_token_usage, record_token_usage, merge_token_usage


def _initial_population_eoh(llm, evaluation, info, pop_size, selection_num=2,
                            num_samplers=1, debug=False, on_evaluated=None,
                            reserve_sample_order=None,
                            initial_sample_nums_max=None):
    """Initialize with the original EoH i1 prompt and greedy survival."""
    sampler = EoHSampler(llm, info["template_program"])
    evaluator = SecureEvaluator(evaluation, debug_mode=debug)
    prompt = EoHPrompt.get_prompt_i1(info)
    print("===== EoH Initialization Prompt =====", flush=True)
    print(prompt, flush=True)
    print("===== End EoH Initialization Prompt =====", flush=True)
    capture_token_usage, record_token_usage, merge_token_usage = (
        _token_usage_helpers(llm))
    target = int(pop_size)
    if initial_sample_nums_max is None:
        initial_sample_nums_max = 2 * target
    initial_sample_nums_max = int(initial_sample_nums_max)
    individuals = []
    attempts_started = 0
    stop = threading.Event()
    lock = threading.RLock()

    def sample_one():
        sample_order = (reserve_sample_order()
                        if reserve_sample_order is not None else None)
        sample_start = time.time()
        evolution_usage = None
        token_usage = None
        try:
            with capture_token_usage() as evolution_usage:
                thought, function = sampler.get_thought_and_function(prompt)
        except Exception as exc:
            if debug:
                print(f"DEBUG: EoH initialization sampling failed: {exc}")
            return None
        finally:
            token_usage = record_token_usage(evolution_usage)
        sample_time = time.time() - sample_start
        if thought is None or function is None:
            return None
        try:
            program = TextFunctionProgramConverter.function_to_program(
                function, info["template_program"])
        except Exception as exc:
            if debug:
                print(f"DEBUG: EoH initialization conversion failed: {exc}")
            program = None
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

    def worker():
        nonlocal attempts_started
        while not stop.is_set():
            with lock:
                if len(individuals) >= target:
                    stop.set()
                    return
                if attempts_started >= initial_sample_nums_max:
                    stop.set()
                    return
                attempts_started += 1
            try:
                function = sample_one()
            except SampleBudgetExhausted:
                stop.set()
                return
            if function is None:
                continue
            with lock:
                if len(individuals) >= target:
                    continue
                if any(str(existing) == str(function) for existing in individuals):
                    continue
                individuals.append(function)
                print(f"EoH init individual {len(individuals)}/{target}, "
                      f"score={function.score}", flush=True)
                if len(individuals) >= target:
                    stop.set()

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(num_samplers))) as executor:
        futures = [executor.submit(worker)
                   for _ in range(max(1, int(num_samplers)))]
        for future in futures:
            future.result()

    individuals.sort(key=lambda item: float(item.score), reverse=True)
    if len(individuals) < selection_num:
        raise RuntimeError(
            f"EoH initialization retained only {len(individuals)} feasible "
            f"algorithms after {attempts_started}/{initial_sample_nums_max} "
            f"samples; at least {selection_num} are required")
    if len(individuals) < pop_size:
        print(f"EoH initialization retained {len(individuals)}/{pop_size} "
              f"algorithms after {attempts_started} samples", flush=True)
    return individuals[:target], merge_token_usage()


def _initial_population(llm, evaluation, info, pop_size, selection_num=2,
                        num_samplers=1, debug=False, on_evaluated=None,
                        reserve_sample_order=None, init_pop_size=None):
    """Run RefineEvo's complete initialization pipeline.

    RefineEvo does not immediately accept the first ``pop_size`` feasible
    algorithms.  It first generates ``init_pop_size`` candidates, evaluates
    all of them, removes invalid candidates and duplicate objectives, and only
    then keeps the best ``pop_size`` candidates.  The MCTS implementation uses
    maximization scores, so RefineEvo's ``nsmallest`` objective selection is
    represented here by ``nlargest`` score selection.
    """
    evaluator = SecureEvaluator(evaluation, debug_mode=debug)
    feasible_individuals = []
    if init_pop_size is None:
        init_pop_size = 3 * int(pop_size)
    init_pop_size = int(init_pop_size)
    if init_pop_size < int(pop_size):
        raise ValueError(
            f"init_pop_size ({init_pop_size}) must be >= pop_size ({pop_size})")
    # RefineEvo population_init() creates floor(init_pop_size/pop_size)
    # complete batches; it does not create a partial final batch.
    n_init_batches = init_pop_size // int(pop_size)
    candidate_target = n_init_batches * int(pop_size)
    if candidate_target != init_pop_size:
        print(
            f"RefineEvo init_pop_size={init_pop_size} is not divisible by "
            f"pop_size={pop_size}; generating {candidate_target} candidates",
            flush=True,
        )
    init_pop_size = candidate_target
    prompt = _refineevo_initial_prompt(info, info["template_module"])
    print("===== Initialization Prompt =====", flush=True)
    print(json.dumps(prompt, ensure_ascii=False, indent=2), flush=True)
    print("===== End Initialization Prompt =====", flush=True)
    capture_token_usage, record_token_usage, merge_token_usage = (
        _token_usage_helpers(llm))

    def sample_one():
        # RefineEvo's EvolutionInterface.get_offspring() retries generation
        # and parsing failures up to five times.  Evaluation failures are not
        # retried: the resulting candidate is simply invalid and is removed by
        # population management.
        last_error = None
        for _ in range(5):
            sample_order = (reserve_sample_order()
                            if reserve_sample_order is not None else None)
            sample_start = time.time()
            evolution_usage = None
            token_usage = None
            try:
                with capture_token_usage() as evolution_usage:
                    thought, function = sample_initial_algorithm(
                        llm, prompt, info["template_program"])
            except Exception as exc:
                last_error = exc
                thought, function = None, None
            finally:
                token_usage = record_token_usage(evolution_usage)
            sample_time = time.time() - sample_start
            if thought is None or function is None:
                continue
            try:
                program = TextFunctionProgramConverter.function_to_program(
                    function, info["template_program"])
            except Exception as exc:
                last_error = exc
                program = None
            if program is None:
                continue
            score, eval_time = evaluator.evaluate_program_record_time(program)
            if score is None or not isinstance(score, (int, float)) \
                    or not math.isfinite(float(score)):
                if debug and score is not None:
                    print(f"DEBUG: rejected non-finite initialization score: {score}")
                return None
            # RefineEvo rounds objectives before population management.  Keep
            # the same precision before score conversion/duplicate filtering.
            function.algorithm = thought
            function.score = float(np.round(score, 5))
            function.evaluate_time = eval_time
            function.sample_time = sample_time
            function._recipe_token_usage = token_usage
            function.operator = "i1"
            function._recipe_sample_order = sample_order
            if on_evaluated is not None:
                on_evaluated(function)
            return function
        if debug and last_error is not None:
            print(f"DEBUG: RefineEvo initialization generation failed after 5 retries: "
                  f"{last_error}")
        return None

    # RefineEvo's population_init() creates complete batches of pop_size
    # (three batches with its default 30/10 configuration). A failed
    # parse/evaluation remains an invalid candidate and is filtered by the
    # management step below; it is not replaced by an extra sample.
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, num_samplers)) as executor:
        sample_limit_reached = False
        for batch_start in range(0, init_pop_size, int(pop_size)):
            if sample_limit_reached:
                break
            batch_end = min(batch_start + int(pop_size), init_pop_size)
            for offset in range(batch_start, batch_end, max(1, num_samplers)):
                if sample_limit_reached:
                    break
                batch_size = min(max(1, num_samplers), batch_end - offset)
                try:
                    for function in executor.map(
                            lambda _: sample_one(), range(batch_size)):
                        if function is not None:
                            feasible_individuals.append(function)
                            print(f"RefineEvo init candidate "
                                  f"{len(feasible_individuals)}/{init_pop_size}, "
                                  f"score={function.score}", flush=True)
                except SampleBudgetExhausted:
                    sample_limit_reached = True
                    break
        if sample_limit_reached:
            print("Initialization stopped because the sample limit was reached.",
                  flush=True)

    # Equivalent to RefineEvo PopulationManagement.pop_greedy(): discard
    # invalid candidates, keep one candidate per objective, then retain the
    # best population size.  LLM4AD scores are maximized.
    # RefineEvo's pop_greedy() first keeps the first individual encountered
    # for each objective, then selects the best remaining objectives.
    unique = []
    seen_scores = set()
    for function in feasible_individuals:
        score_key = float(function.score)
        if score_key in seen_scores:
            continue
        seen_scores.add(score_key)
        unique.append(function)
    individuals = sorted(
        unique,
        key=lambda item: float(item.score),
        reverse=True,
    )[:int(pop_size)]

    print(
        f"RefineEvo initialization generated {init_pop_size} candidates, "
        f"{len(feasible_individuals)} feasible, "
        f"{len(individuals)} retained after population management",
        flush=True,
    )
    if len(individuals) < selection_num:
        raise RuntimeError(
            f"RefineEvo initialization retained only {len(individuals)} unique "
            f"algorithms from {init_pop_size} candidates; at least "
            f"{selection_num} are required")
    if len(individuals) < pop_size:
        print(f"RefineEvo initialization retained "
              f"{len(individuals)}/{pop_size} unique algorithms", flush=True)
    return individuals, merge_token_usage()


def run_task(*, method_name, template_module, evaluation, llm,
             store_dir, pop_size=10, selection_num=2, max_depth=50,
             max_sample_count=10000, num_samplers=10, num_evaluators=10,
             init_pop_size=30, debug=False, elite_pool_size=0,
             checkpoint=None, mode="reflection", operators=None,
             embedding_config=None, exploration_constant=0.1,
             depth_balance_weight=0.2, node_batch_size=10, seed=None,
             experiment_config=None, initialization_mode="refineevo",
             initial_sample_nums_max=None):
    from example.tasks.utils import get_info

    info = get_info(method_name, template_module)
    info["template_module"] = template_module
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
    mode = str(mode or "reflection").strip().lower()
    if mode not in {"reflection", "no_reflection"}:
        raise ValueError(
            "mcts recipe mode must be 'reflection' or 'no_reflection'")
    initialization_mode = str(initialization_mode or "refineevo").lower()
    if initialization_mode not in {"refineevo", "eoh", "original"}:
        raise ValueError(
            "initialization_mode must be 'refineevo' or 'eoh'")
    print(f"Recipe-MCTS mode: {mode}", flush=True)
    print(f"Recipe-MCTS initialization mode: {initialization_mode}", flush=True)
    print(f"Recipe-MCTS max samples: {max_sample_count}", flush=True)
    elite_pool_size = int(elite_pool_size or 0)
    if elite_pool_size > 0:
        print(f"Recipe-MCTS global elite pool size: {elite_pool_size}",
              flush=True)
    config_snapshot_path = write_experiment_config_snapshot(
        store_dir,
        experiment_config=experiment_config,
        method_name=method_name,
        template_module=template_module,
        evaluation=evaluation,
        resolved_options={
            "store_dir": store_dir,
            "checkpoint": checkpoint,
            "mode": mode,
            "operators": operators,
            "pop_size": pop_size,
            "selection_num": selection_num,
            "max_depth": max_depth,
            "max_sample_count": max_sample_count,
            "initialization_mode": initialization_mode,
            "initial_sample_nums_max": initial_sample_nums_max,
            "init_pop_size": init_pop_size,
            "num_samplers": num_samplers,
            "num_evaluators": num_evaluators,
            "debug": debug,
            "elite_pool_size": elite_pool_size,
            "embedding_config": embedding_config,
            "exploration_constant": exploration_constant,
            "depth_balance_weight": depth_balance_weight,
            "node_batch_size": node_batch_size,
            "seed": seed,
        },
        recipes=recipes,
    )
    print(f"Experiment config snapshot: {config_snapshot_path}", flush=True)

    experience_manager = None
    if mode == "reflection":
        embedding_config = dict(embedding_config or {})
        embedding_host = embedding_config.get(
            "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        embedding = OpenAIEmbedding(
            api_key=_secret(embedding_config, default=""),
            base_url=embedding_host,
            model=embedding_config.get("model", "text-embedding-v4"),
            encoding_format=embedding_config.get("encoding_format", "float"),
        )
        experience_manager = RefineEvoExperienceManager(
            reflector_llm=llm,
            embedding_model=embedding,
            top_k=int(embedding_config.get(
                "top_k", recipes["refineevo_experience"]["experience_top_k"])),
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
            operators=operators,
            **kwargs,
        )
    method = MCTSRecipe(
        recipes=recipes,
        pop_size=pop_size,
        selection_num=selection_num,
        max_depth=max_depth,
        exploration_constant=float(exploration_constant),
        depth_balance_weight=float(depth_balance_weight),
        node_batch_size=int(node_batch_size),
        store_dir=str(store_dir),
        expand_fn=expanders,
        seed=(None if seed is None else int(seed)),
        max_sample_count=max_sample_count,
        elite_pool_size=elite_pool_size,
    )
    try:
        if checkpoint:
            initial = None
        else:
            # Construct the recorder before sampling so every evaluated root
            # candidate can update sample_best.json immediately.
            init_callback = lambda function: method.record_evaluated_sample(
                function, population_node_id=1, depth=0)
            if initialization_mode in {"eoh", "original"}:
                initial, initial_token_usage = _initial_population_eoh(
                    llm, evaluation, info, pop_size, selection_num,
                    num_samplers, debug,
                    on_evaluated=init_callback,
                    reserve_sample_order=method.reserve_sample_order,
                    initial_sample_nums_max=initial_sample_nums_max,
                )
            else:
                initial, initial_token_usage = _initial_population(
                    llm, evaluation, info, pop_size, selection_num,
                    num_samplers, debug,
                    on_evaluated=init_callback,
                    reserve_sample_order=method.reserve_sample_order,
                    init_pop_size=init_pop_size,
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


def common_options(default_log, config=None):
    config = config or {}
    mcts_config = _section(config, "mcts")
    repository_root = _repo_root()
    configured_log = config.get("log_dir", default_log)
    store_dir = Path(configured_log)
    if not store_dir.is_absolute():
        store_dir = repository_root / store_dir
    pop_size = int(mcts_config.get("pop_size", 10))
    num_samplers = mcts_config.get("num_samplers", pop_size)
    num_evaluators = mcts_config.get("num_evaluators", pop_size)
    elite_pool_size = int(mcts_config.get("elite_pool_size", 0) or 0)
    return {
        "store_dir": store_dir,
        "pop_size": pop_size,
        "selection_num": int(mcts_config.get("selection_num", 2)),
        "max_depth": int(mcts_config.get("max_depth", 50)),
        "max_sample_count": int(mcts_config.get("max_sample_count", 10000)),
        "initialization_mode": mcts_config.get("initialization_mode", "refineevo"),
        "initial_sample_nums_max": mcts_config.get("initial_sample_nums_max"),
        # RefineEvo's default is to generate 30 initial candidates and retain
        # the best 10 for the initial population.
        "init_pop_size": int(mcts_config.get("init_pop_size", 30)),
        # Match EoH-style generation by default: one concurrent sampler and
        # evaluator pipeline per population slot. Both values remain
        # independently configurable for resource-constrained machines.
        "num_samplers": int(num_samplers),
        "num_evaluators": int(num_evaluators),
        "debug": bool(mcts_config.get("debug", False)),
        "elite_pool_size": elite_pool_size,
        "checkpoint": config.get("checkpoint"),
        "mode": ("no_reflection" if config.get("no_reflection", False)
                 else config.get("mode", "reflection")),
        "operators": _operators(config),
        "embedding_config": _section(config, "embedding"),
        "exploration_constant": float(mcts_config.get("exploration_constant", 0.1)),
        "depth_balance_weight": float(mcts_config.get("depth_balance_weight", 0.2)),
        "node_batch_size": int(mcts_config.get("node_batch_size", 10)),
        "seed": mcts_config.get("seed"),
        "experiment_config": config,
    }
