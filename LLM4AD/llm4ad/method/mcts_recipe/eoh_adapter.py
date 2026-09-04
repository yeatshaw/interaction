"""Adapter between Recipe-MCTS and the existing EoH prompt/evaluation stack."""

from __future__ import annotations

import copy
import concurrent.futures
import math
import threading
import time

from ..eoh.prompt import EoHPrompt
from ..eoh.sampler import EoHSampler
from ...base import TextFunctionProgramConverter, SecureEvaluator


class EoHRecipeExpander:
    """Generate one feasible EoH-sized offspring batch for a recipe.

    The operator schedule is intentionally the EoH schedule (e1, e2, m1, m2).
    ``recipe`` only changes reflection flags and input options.
    """

    def __init__(self, llm, evaluation, info, template_program,
                 num_samplers=10, num_evaluators=1, debug_mode=False,
                 **evaluator_kwargs):
        self.info = dict(info)
        self.template_program = template_program
        self.sampler = EoHSampler(llm, template_program)
        self.evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode,
                                         **evaluator_kwargs)
        self.num_samplers = max(1, int(num_samplers))
        self.num_evaluators = max(1, int(num_evaluators))
        # Sampling and evaluation are deliberately separate pools.  A sampling
        # worker never waits for evaluation; the whole generated batch is
        # submitted to ``evaluation_executor`` after all sampling futures have
        # completed.
        self.sampler_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_samplers)
        self.evaluation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_evaluators)
        # Keep the old attribute for callers that inspected the adapter.
        self.executor = self.evaluation_executor
        self._prompt_lock = threading.RLock()
        self._closed = False
        self.debug_mode = debug_mode

    def close(self, close_llm=True):
        if self._closed:
            return
        self._closed = True
        self.sampler_executor.shutdown(wait=True, cancel_futures=True)
        self.evaluation_executor.shutdown(wait=True, cancel_futures=True)
        if close_llm:
            self.sampler.llm.close()

    def _reflect(self, children, population, recipe, parents=None):
        values = recipe.values
        parents = parents or [self._parents(children, ref) for ref in children]
        prompt = EoHPrompt.get_prompt_reflection(
            children, parents=parents, info=self.info,
            parent_info_flag=values.get("parent_info", True),
            best_worst_flag=values.get("best_worst", False),
            fitness_flag=values.get("fitness", 2),
            avg_fitness_flag=values.get("avg_fitness", False),
            check_reflection_flag=values.get("check_guidance", True),
            population=population,
            use_long_term_reflection=False,
            identical_parent_children_flag=values.get("identical_parents", False),
            shared_parent_children_flag=values.get("shared_parent", False),
            population_comparison=values.get("population_comparison"),
            comparison_flag=values.get("comparison", False),
            attribution_flag=values.get("attribution", False),
            summarization_flag=values.get("summarization", False),
            attribution_task=values.get("attribution_type", "good"),
            summarization_task=values.get("summarization_type", "guidance"),
        )
        result = self.sampler.llm.draw_sample(prompt)
        return result.strip() if isinstance(result, str) else None

    @staticmethod
    def _parents(refs, ref):
        # Parent lineage is task-owned. If unavailable, the prompt naturally
        # falls back to the reference-only section.
        return list(getattr(ref, "_recipe_parent_functions", ()))

    def _generation_prompt(self, refs, suggestion, operator):
        """Build an operator prompt without serializing LLM calls.

        ``EoHPrompt`` clears the copied Functions' docstrings while formatting
        them.  The lock protects that formatter, while the actual LLM request
        remains outside the lock and therefore runs in parallel.
        """
        with self._prompt_lock:
            return self._generation_prompt_unlocked(refs, suggestion, operator)

    def _generation_prompt_unlocked(self, refs, suggestion, operator):
        if operator == "e1":
            prompt = EoHPrompt.get_prompt_e1(refs, self.info, suggestion)
        elif operator == "e2":
            prompt = EoHPrompt.get_prompt_e2(refs, self.info, suggestion)
        elif operator == "m1":
            prompt = EoHPrompt.get_prompt_m1(refs[0], self.info, suggestion)
        else:
            prompt = EoHPrompt.get_prompt_m2(refs[0], self.info, suggestion)
        return prompt

    def _prepare_candidate(self, refs, population, recipe, operator):
        """Perform reflection and LLM/code generation, but do not evaluate.

        The returned dictionary is intentionally independent of evaluation so
        many of these tasks can be generated concurrently and then evaluated
        as one batch.
        """
        sample_start = time.time()
        try:
            # Front-loaded reflection. References without lineage parents use
            # the reference section; later generations use parents vs children.
            suggestion = self._reflect(refs, population, recipe)
            prompt = self._generation_prompt(refs, suggestion, operator)
            thought, function = self.sampler.get_thought_and_function(prompt)
            if thought is None or function is None:
                return None
            program = TextFunctionProgramConverter.function_to_program(
                function, self.template_program)
            if program is None:
                return None
        except Exception as exc:
            if self.debug_mode:
                print(f"DEBUG: candidate generation failed: {exc}")
            return None

        sample_time = time.time() - sample_start
        function.algorithm = thought
        function.sample_time = sample_time
        function.operator = operator
        function._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        function._recipe_id = recipe.recipe_id
        function._eoh_generation_suggestion = suggestion
        return {
            "function": function,
            "program": program,
            "refs": refs,
            "operator": operator,
            "suggestion": suggestion,
            "sample_time": sample_time,
        }

    def _evaluate_candidates(self, candidates):
        """Evaluate a prepared batch concurrently and return feasible entries."""
        if not candidates:
            return []
        futures = {
            self.evaluation_executor.submit(
                self.evaluator.evaluate_program_record_time,
                candidate["program"]): candidate
            for candidate in candidates
        }
        evaluated = []
        for future, candidate in futures.items():
            try:
                result = future.result()
                if not isinstance(result, tuple) or len(result) != 2:
                    score, eval_time = None, None
                else:
                    score, eval_time = result
            except Exception as exc:
                if self.debug_mode:
                    print(f"DEBUG: candidate evaluation failed: {exc}")
                score, eval_time = None, None
            if score is None or not isinstance(score, (int, float)) \
                    or not math.isfinite(float(score)):
                if self.debug_mode and score is not None:
                    print(f"DEBUG: rejected non-finite evaluation score: {score}")
                continue
            function = candidate["function"]
            function.score = score
            function.evaluate_time = eval_time
            evaluated.append(candidate)
        return evaluated

    def _generate(self, refs, population, recipe, operator):
        """Compatibility wrapper for one candidate through the two-stage path."""
        candidate = self._prepare_candidate(
            copy.deepcopy(list(refs)), population, recipe, operator)
        evaluated = self._evaluate_candidates([candidate] if candidate else [])
        if not evaluated:
            return None
        function = evaluated[0]["function"]
        return function

    def __call__(self, population, recipe, selection_num, target_size):
        operators = ("e1", "e2", "m1", "m2")
        # Keep a bad/invalid LLM response from spinning forever.  This is the
        # same two-sample-per-slot budget used by EoH initialization; a short
        # batch simply causes the MCTS caller to inherit the parent population.
        max_attempts = max(int(target_size) * 2, int(target_size))
        specs = []
        for index in range(max_attempts):
            operator = operators[index % len(operators)]
            try:
                count = selection_num if operator in ("e1", "e2") else 1
                refs = copy.deepcopy(population.select_many(count))
            except Exception as exc:
                if self.debug_mode:
                    print(f"DEBUG: reference selection failed: {exc}")
                refs = None
            if refs:
                specs.append((refs, population, recipe, operator))

        # Node-level barrier: every LLM/code-generation future completes
        # before the first evaluation future is submitted.
        prepared = list(self.sampler_executor.map(
            lambda spec: self._prepare_candidate(*spec), specs))
        evaluated = self._evaluate_candidates(
            [candidate for candidate in prepared if candidate is not None])
        offspring = [candidate["function"] for candidate in evaluated[:target_size]]
        if len(offspring) < target_size and self.debug_mode:
            print(f"DEBUG: recipe {recipe.recipe_id} produced "
                  f"{len(offspring)}/{target_size} feasible candidates "
                  f"after {len(specs)} attempts")
        return offspring

    def _generate_without_reflection(self, refs, recipe, operator="e1"):
        """Generate one child using the original EoH operator prompt."""
        try:
            refs = copy.deepcopy(list(refs))
            prompt = self._generation_prompt(refs, None, operator)
            sample_start = time.time()
            thought, function = self.sampler.get_thought_and_function(prompt)
            if thought is None or function is None:
                return None
            program = TextFunctionProgramConverter.function_to_program(
                function, self.template_program)
            if program is None:
                return None
            function.algorithm = thought
            function.sample_time = time.time() - sample_start
            function.operator = operator
            function._recipe_id = recipe.recipe_id
            function._eoh_parent_ids = tuple(
                getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
            candidate = {"function": function, "program": program,
                         "refs": refs, "operator": operator}
            evaluated = self._evaluate_candidates([candidate])
            return evaluated[0]["function"] if evaluated else None
        except Exception as exc:
            if self.debug_mode:
                print(f"DEBUG: candidate generation failed: {exc}")
            return None

    # The implementation below is retained for compatibility with code that
    # directly called the old methods.  Normal node expansion uses ``__call__``
    # and the strict two-stage batch path above.

class RefineEvoRecipeExpander(EoHRecipeExpander):
    """Front-loaded RefineEvo-style experience retrieval and reflection.

    The selected reference is the child in an already completed historical
    parent-to-child trajectory. If lineage parents exist, distillation compares
    them; otherwise it falls back to reference-only reflection. Retrieved and
    newly distilled experience then guides generation of the next child.

    ``retrieve_experiences`` and ``distill_experience`` are injected so the
    node-local embedding store remains independent of this adapter.
    """

    def __init__(self, *args, retrieve_experiences=None,
                 distill_experience=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieve_experiences = retrieve_experiences
        self.distill_experience = distill_experience

    @staticmethod
    def _objective_improved(child, parents):
        scores = [getattr(x, "score", None) for x in parents]
        scores = [float(x) for x in scores if x is not None]
        child_score = getattr(child, "score", None)
        # EoH maximizes score (negative cost), so improvement is larger score.
        return child_score is not None and (not scores or child_score > max(scores))

    def _generate_without_reflection(self, refs, recipe):
        """Compatibility wrapper for the old no-experience call site."""
        operator = getattr(self, "_next_operator", "e1")
        function = super()._generate_without_reflection(refs, recipe, operator)
        return function, operator

    def _prepare_experience_candidate(self, refs, recipe, operator,
                                      parent_experiences):
        """Retrieve/distill experience and generate one candidate.

        This is the sampling phase of RefineEvo.  Distillation may itself call
        the reflector LLM, so it stays in the sampler pool; no evaluation is
        started until every candidate in the batch has reached this method's
        return point.
        """
        sample_start = time.time()
        try:
            retrieved = (self.retrieve_experiences(
                refs, operator, list(parent_experiences))
                if self.retrieve_experiences else [])
            retrieved = retrieved if isinstance(retrieved, list) else []
            fresh = (self.distill_experience(refs, operator)
                     if self.distill_experience else [])
            fresh = fresh if isinstance(fresh, list) else ([fresh] if fresh else [])
            guidance = self._format_experiences(retrieved + fresh)
            prompt = self._generation_prompt(refs, guidance or None, operator)
            if guidance:
                prompt = prompt.replace(
                    "These are some suggestions after reflecting on the given algorithms:",
                    "These are successful and failed design experiences retrieved from previous attempts:")
            thought, function = self.sampler.get_thought_and_function(prompt)
            if thought is None or function is None:
                return None
            program = TextFunctionProgramConverter.function_to_program(
                function, self.template_program)
            if program is None:
                return None
        except Exception as exc:
            if self.debug_mode:
                print(f"DEBUG: RefineEvo candidate generation failed: {exc}")
            return None

        sample_time = time.time() - sample_start
        function.algorithm = thought
        function.sample_time = sample_time
        function.operator = operator
        function._recipe_id = recipe.recipe_id
        function._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        # Preserve the experience actually supplied to the operator on the
        # individual as well as in the node-level experience pool.
        function._eoh_experience = guidance or None
        return {
            "function": function,
            "program": program,
            "refs": refs,
            "operator": operator,
            "retrieved": retrieved,
            "fresh": fresh,
            "experience_text": guidance,
            "sample_time": sample_time,
        }

    def __call__(self, population, recipe, selection_num, target_size,
                 experiences=None):
        parent_experiences = list(experiences or [])
        experience_lock = threading.RLock()
        manager = getattr(self.retrieve_experiences, "__self__", None)
        if manager is not None and hasattr(manager, "begin_node"):
            manager.begin_node(parent_experiences)
        operators = ("e1", "e2", "m1", "m2")
        max_attempts = max(int(target_size) * 2, int(target_size))

        try:
            # Freeze one node-local experience snapshot for the complete
            # generation phase.  No evaluation or score update happens yet.
            with experience_lock:
                experience_snapshot = copy.deepcopy(parent_experiences)
            specs = []
            for index in range(max_attempts):
                operator = operators[index % len(operators)]
                try:
                    count = selection_num if operator in ("e1", "e2") else 1
                    refs = copy.deepcopy(population.select_many(count))
                except Exception as exc:
                    if self.debug_mode:
                        print(f"DEBUG: reference selection failed: {exc}")
                    refs = None
                if refs:
                    specs.append((refs, recipe, operator, experience_snapshot))
            prepared = list(self.sampler_executor.map(
                lambda spec: self._prepare_experience_candidate(*spec), specs))
            evaluated = self._evaluate_candidates(
                [candidate for candidate in prepared if candidate is not None])

            # The node-local experience pool is mutated only after the whole
            # candidate batch has been evaluated.
            offspring, new_experiences = [], []
            with experience_lock:
                for candidate in evaluated[:target_size]:
                    child = candidate["function"]
                    refs = candidate["refs"]
                    retrieved = candidate["retrieved"]
                    fresh = candidate["fresh"]
                    offspring.append(child)
                    improved = self._objective_improved(child, refs)
                    retrieved_ids = {
                        item.get("experience_id") for item in retrieved
                        if isinstance(item, dict)
                    }
                    retained = []
                    for item in parent_experiences:
                        if not isinstance(item, dict):
                            retained.append(item)
                            continue
                        if item.get("experience_id") in retrieved_ids:
                            item["score"] = int(item.get("score", 0)) + (
                                1 if improved else -1)
                        if int(item.get("score", 0)) >= 0:
                            retained.append(item)
                    parent_experiences[:] = retained
                    new_experiences.extend(copy.deepcopy(fresh))
        finally:
            if isinstance(experiences, list):
                experiences[:] = parent_experiences
            if manager is not None and hasattr(manager, "end_node"):
                manager.end_node()
        if len(offspring) < target_size and self.debug_mode:
            print(f"DEBUG: recipe {recipe.recipe_id} produced "
                  f"{len(offspring)}/{target_size} feasible candidates "
                  f"after {len(specs)} attempts")
        return offspring, new_experiences

    def _generate_with_experience(self, refs, recipe, experience_text,
                                  operator=None):
        """Compatibility wrapper through the prepared/evaluated path."""
        operator = operator or getattr(self, "_next_operator", "e1")
        candidate = self._prepare_experience_candidate(
            copy.deepcopy(list(refs)), recipe, operator, [])
        if candidate is None:
            return None, operator
        # The compatibility argument is already formatted text, while the
        # normal path retrieves it inside ``_prepare_experience_candidate``.
        if experience_text:
            candidate["function"]._eoh_experience = experience_text
        evaluated = self._evaluate_candidates([candidate])
        if not evaluated:
            return None, operator
        return evaluated[0]["function"], operator

    @staticmethod
    def _format_experiences(experiences):
        if not experiences:
            return ""
        success, failure, reference = [], [], []
        for item in experiences:
            text = item if isinstance(item, str) else str(item)
            status = item.get("is_success") if isinstance(item, dict) else None
            (success if status is True else failure if status is False else reference).append(text)
        return ("### DESIGN EXPERIENCES FROM PREVIOUS ATTEMPTS\n"
                "#### SUCCESSFUL EXPERIENCES\n" + "\n".join(success or ["None"]) + "\n"
                "#### FAILED EXPERIENCES\n" + "\n".join(failure or ["None"]) + "\n"
                "#### REFERENCE-ONLY EXPERIENCES\n" + "\n".join(reference or ["None"]))

    @staticmethod
    def _next_operator_after(operator):
        order = ("e1", "e2", "m1", "m2")
        return order[(order.index(operator) + 1) % len(order)]
