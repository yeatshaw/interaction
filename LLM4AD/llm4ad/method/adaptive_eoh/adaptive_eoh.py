from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import Counter

from ..eoh.eoh import EoH
from ..eoh.profiler import EoHProfiler
from ..eoh.prompt import EoHPrompt
from ...base import Function, TextFunctionProgramConverter
from ...method.mcts_recipe.recipes import get_default_recipes


class AdaptiveRecipeEoH(EoH):
    """Single-path EoH where every offspring chooses its own reflection recipe.

    The parent EoH and MCTS-Recipe implementations are deliberately untouched.
    Recipe prompts are produced by the existing ``EoHPrompt`` implementation;
    this class only chooses which existing configuration is used for one sample.
    ``recipe_reflectors`` can provide a task-specific implementation for the
    RefineEvo recipe.  Other recipes use the normal EoH reflection prompt.
    """

    def __init__(self, *args, recipe_configs=None, recipe_temperature=1.0,
                 recipe_reflectors=None, refineevo_manager=None,
                 recipe_stats_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._recipe_configs = recipe_configs or get_default_recipes()
        if not self._recipe_configs:
            raise ValueError("recipe_configs must contain at least one recipe")
        self._recipe_temperature = float(recipe_temperature)
        if self._recipe_temperature <= 0:
            raise ValueError("recipe_temperature must be positive")
        self._recipe_reflectors = dict(recipe_reflectors or {})
        if refineevo_manager is not None:
            from .refineevo import AdaptiveRefineEvoReflector
            for name, config in self._recipe_configs.items():
                if config.get("refineevo_experience", False):
                    self._recipe_reflectors.setdefault(
                        name, AdaptiveRefineEvoReflector(
                            refineevo_manager, self.get_lineage_parents))
        missing = [name for name, config in self._recipe_configs.items()
                   if config.get("refineevo_experience", False)
                   and name not in self._recipe_reflectors]
        if missing:
            raise ValueError(
                "RefineEvo recipes require recipe_reflectors callbacks: "
                + ", ".join(missing))
        self._recipe_lock = threading.RLock()
        self._sample_lock = threading.RLock()
        self._prompt_lock = threading.RLock()
        self._recipe_local = threading.local()
        self._recipe_gain = {name: 0.0 for name in self._recipe_configs}
        self._recipe_count = Counter()
        self._recipe_stats_path = recipe_stats_path
        if self._recipe_stats_path is None and self._profiler is not None:
            self._recipe_stats_path = os.path.join(
                self._profiler._log_dir, "adaptive_recipe_stats.json")
        if self._recipe_stats_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._recipe_stats_path)),
                        exist_ok=True)

    def _recipe_probabilities(self):
        with self._recipe_lock:
            names = list(self._recipe_configs)
            values = [self._recipe_gain[name] / self._recipe_temperature
                      for name in names]
        shift = max(values)
        weights = [math.exp(value - shift) for value in values]
        total = sum(weights)
        return names, [weight / total for weight in weights]

    @staticmethod
    def _function_from_lineage_record(record):
        parent = Function(
            name=record["name"],
            args=record["args"],
            body=record["body"],
            return_type=record.get("return_type"),
            score=record.get("score"),
        )
        parent.algorithm = record.get("algorithm", "")
        parent.operator = record.get("operator")
        parent._eoh_generation_suggestion = record.get("suggestion")
        parent._eoh_experience = record.get("experience")
        parent._eoh_lineage_id = record["node_id"]
        parent._eoh_parent_ids = tuple(record.get("parent_ids", ()))
        return parent

    def get_lineage_parents(self, func):
        """Resolve parents from both the pending cache and persisted shards.

        The original EoH lookup intentionally remains unchanged. Adaptive
        recipes need the newest lineage immediately because reflection can use
        an offspring before the current 100-record shard has been flushed.
        """
        parent_ids = tuple(getattr(func, "_eoh_parent_ids", ()))
        if not parent_ids:
            return []

        records = {}
        with self._lineage_lock:
            pending = {
                int(record["node_id"]): record
                for record in self._lineage_cache
            }
            for parent_id in parent_ids:
                numeric_id = int(parent_id)
                if numeric_id in pending:
                    records[numeric_id] = pending[numeric_id]
                    continue
                path = self._lineage_shard_path(numeric_id)
                try:
                    with open(path, encoding="utf-8") as file:
                        record = next(
                            (node for node in json.load(file).get("nodes", ())
                             if int(node["node_id"]) == numeric_id),
                            None,
                        )
                except (FileNotFoundError, json.JSONDecodeError):
                    record = None
                if record is not None:
                    records[numeric_id] = record

        return [
            self._function_from_lineage_record(records[int(parent_id)])
            for parent_id in parent_ids
            if int(parent_id) in records
        ]

    def _select_recipe(self):
        names, probabilities = self._recipe_probabilities()
        recipe_id = random.choices(names, weights=probabilities, k=1)[0]
        return recipe_id, probabilities[names.index(recipe_id)]

    def _recipe_reflection(self, refs, recipe_id, operator):
        config = self._recipe_configs[recipe_id]
        callback = self._recipe_reflectors.get(recipe_id)
        if callback is not None:
            return callback(refs, self._population, config, operator)

        # All non-RefineEvo recipes are exactly the existing EoH prompt with
        # different flags.  No prompt text or section is reimplemented here.
        return self._reflect_with_config(refs, config)

    def _reflect_with_config(self, refs, config):
        lineage_parents = [self.get_lineage_parents(child) for child in refs]
        with self._prompt_lock:
            prompt = EoHPrompt.get_prompt_reflection(
                refs,
                parents=lineage_parents,
                info=self._info,
                parent_info_flag=config.get("parent_info", False),
                best_worst_flag=config.get("best_worst", False),
                fitness_flag=config.get("fitness", 0),
                avg_fitness_flag=config.get("avg_fitness", False),
                check_reflection_flag=config.get("check_guidance", False),
                population=self._population,
                use_long_term_reflection=config.get(
                    "use_long_term_reflection", False),
                identical_parent_children_flag=config.get(
                    "identical_parents", False),
                shared_parent_children_flag=config.get("shared_parent", False),
                population_comparison=config.get("population_comparison"),
                comparison_flag=config.get("comparison", False),
                attribution_flag=config.get("attribution", False),
                summarization_flag=config.get("summarization", False),
                attribution_task=config.get("attribution_type", "good"),
                summarization_task=config.get(
                    "summarization_type", "guidance"),
            )
        return self._sampler.llm.draw_sample(prompt).strip()

    def _prepare_reflection(self, refs, operator=None):
        """Choose and execute one recipe for this sample only."""
        if not self._population.population:
            self._recipe_local.context = {
                "recipe_id": None, "probability": 0.0,
                "suggestion": None, "baseline": None,
            }
            return
        recipe_id, probability = self._select_recipe()
        recipe_state = None
        try:
            reflection = self._recipe_reflection(refs, recipe_id, operator)
            if isinstance(reflection, tuple):
                suggestion, recipe_state = reflection
            else:
                suggestion, recipe_state = reflection, None
        except Exception:
            if self._debug_mode:
                import traceback
                traceback.print_exc()
            suggestion = None
        baseline = max((float(ref.score) for ref in refs
                        if ref.score is not None), default=None)
        self._recipe_local.context = {
            "recipe_id": recipe_id,
            "probability": probability,
            "suggestion": suggestion,
            "baseline": baseline,
            "recipe_state": recipe_state,
        }

    def _update_recipe(self, context, score):
        recipe_id = context.get("recipe_id")
        if recipe_id is None or score is None:
            return
        baseline = context.get("baseline")
        gain = max(0.0, float(score) - baseline) if baseline is not None else 0.0
        with self._recipe_lock:
            self._recipe_gain[recipe_id] += gain
            self._recipe_count[recipe_id] += 1
            payload = {
                "cumulative_gain": dict(self._recipe_gain),
                "selection_count": dict(self._recipe_count),
                "mean_gain": {
                    name: (self._recipe_gain[name] / self._recipe_count[name]
                           if self._recipe_count[name] else 0.0)
                    for name in self._recipe_configs
                },
            }
            if self._recipe_stats_path:
                temp = self._recipe_stats_path + ".tmp"
                with open(temp, "w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, indent=2)
                os.replace(temp, self._recipe_stats_path)
        context["gain"] = gain
        callback = self._recipe_reflectors.get(recipe_id)
        if callback is not None and hasattr(callback, "on_result"):
            callback.on_result(context.get("recipe_state"), gain > 0)

    def _sample_evaluate_register(self, prompt, parents=None, operator=None):
        context = getattr(self._recipe_local, "context", {
            "recipe_id": None, "probability": 0.0,
            "suggestion": None, "baseline": None,
        })
        sample_start = time.time()
        thought, function = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or function is None:
            return False
        program = TextFunctionProgramConverter.function_to_program(
            function, self._template_program)
        if program is None:
            return False
        score, evaluate_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time, program).result()

        function.score = score
        function.evaluate_time = evaluate_time
        function.algorithm = thought
        function.sample_time = sample_time
        function.operator = operator
        function.recipe_id = context.get("recipe_id")
        function.recipe_probability = context.get("probability", 0.0)
        function._eoh_generation_suggestion = (
            context.get("suggestion") if operator != "i1" else None)
        function._eoh_experience = None
        self._update_recipe(context, score)
        function.recipe_gain = context.get("gain", 0.0)

        with self._sample_lock:
            self._tot_sample_nums += 1
            if score is not None and (self._max_sample_nums is None or
                    len(self._convergence_history) < self._max_sample_nums):
                self._best_score_so_far = max(self._best_score_so_far, score)
                self._convergence_history.append(
                    (self._tot_sample_nums, self._best_score_so_far))
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
        if score is None:
            return False
        self._register_lineage_node(function, parents)
        self._population.register_function(function)
        self._recipe_local.context = {}
        return True

    def _register_lineage_node(self, function, parents=None):
        """Persist the normal EoH lineage plus adaptive-recipe metadata."""
        super()._register_lineage_node(function, parents)
        path = self._recipe_stats_path
        if not path:
            log_dir = getattr(self._profiler, "_log_dir", None)
            path = (os.path.join(log_dir, "adaptive_recipe_samples.jsonl")
                    if log_dir else None)
        if path:
            sample_path = os.path.join(os.path.dirname(path),
                                       "adaptive_recipe_samples.jsonl")
            record = {
                "algorithm_id": getattr(function, "_eoh_lineage_id", None),
                "generation": self._population.generation,
                "parent_algorithm_ids": list(
                    getattr(function, "_eoh_parent_ids", ())),
                "operator": function.operator,
                "recipe_id": getattr(function, "recipe_id", None),
                "recipe_probability": getattr(
                    function, "recipe_probability", None),
                "recipe_gain": getattr(function, "recipe_gain", None),
                "score": function.score,
                "suggestion": getattr(
                    function, "_eoh_generation_suggestion", None),
                "sample_time": function.sample_time,
                "evaluate_time": function.evaluate_time,
            }
            with self._recipe_lock:
                with open(sample_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _operator_step(self, operator):
        count = self._selection_num if operator in ("e1", "e2") else 1
        refs = (self._population.selection_many(count) if count > 1
                else [self._population.selection()])
        self._prepare_reflection(refs, operator)
        suggestion = self._recipe_local.context.get("suggestion")
        with self._prompt_lock:
            if operator == "e1":
                prompt = EoHPrompt.get_prompt_e1(refs, self._info, suggestion)
            elif operator == "e2":
                prompt = EoHPrompt.get_prompt_e2(refs, self._info, suggestion)
            elif operator == "m1":
                prompt = EoHPrompt.get_prompt_m1(refs[0], self._info, suggestion)
            else:
                prompt = EoHPrompt.get_prompt_m2(refs[0], self._info, suggestion)
        recipe_id = self._recipe_local.context.get("recipe_id")
        if (suggestion and recipe_id is not None and
                self._recipe_configs[recipe_id].get(
                    "refineevo_experience", False)):
            prompt = prompt.replace(
                "These are some suggestions after reflecting on the given algorithms:",
                "These are successful and failed design experiences retrieved "
                "from previous attempts:")
        return self._sample_evaluate_register(prompt, refs, operator)

    def _iteratively_use_eoh_operator(self):
        operators = ["e1"]
        if self._use_e2_operator:
            operators.append("e2")
        if self._use_m1_operator:
            operators.append("m1")
        if self._use_m2_operator:
            operators.append("m2")
        index = 0
        while self._continue_loop():
            try:
                self._operator_step(operators[index % len(operators)])
                index += 1
            except KeyboardInterrupt:
                return
            except Exception:
                if self._debug_mode:
                    import traceback
                    traceback.print_exc()

    def run(self):
        """Use the original EoH lifecycle with the corrected callable handoff."""
        if not self._resume_mode:
            self._multi_threaded_sampling(self._iteratively_init_population)
            self._population.survival()
            if len(self._population) < self._selection_num:
                print("AdaptiveRecipeEoH stopped: too few feasible initial algorithms")
                return
        self._multi_threaded_sampling(self._iteratively_use_eoh_operator)
        self._evaluation_executor.shutdown(wait=True, cancel_futures=True)
        if self._profiler is not None:
            self._profiler.finish()
        with self._lineage_lock:
            self._flush_lineage_cache()
        self._sampler.llm.close()
