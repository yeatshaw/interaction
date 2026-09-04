"""Adapter between Recipe-MCTS and the existing EoH prompt/evaluation stack."""

from __future__ import annotations

import copy
import concurrent.futures
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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        self.num_samplers = max(1, int(num_samplers))
        self.debug_mode = debug_mode

    def close(self, close_llm=True):
        self.executor.shutdown(wait=True, cancel_futures=True)
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

    def _generate(self, refs, population, recipe, operator):
        # Front-loaded reflection. References without lineage parents use the
        # reference section; later generations use parents vs children.
        suggestion = self._reflect(refs, population, recipe)
        if operator == "e1":
            prompt = EoHPrompt.get_prompt_e1(refs, self.info, suggestion)
        elif operator == "e2":
            prompt = EoHPrompt.get_prompt_e2(refs, self.info, suggestion)
        elif operator == "m1":
            prompt = EoHPrompt.get_prompt_m1(refs[0], self.info, suggestion)
        else:
            prompt = EoHPrompt.get_prompt_m2(refs[0], self.info, suggestion)
        sample_start = time.time()
        thought, function = self.sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or function is None:
            return None
        program = TextFunctionProgramConverter.function_to_program(function, self.template_program)
        if program is None:
            return None
        score, eval_time = self.executor.submit(
            self.evaluator.evaluate_program_record_time, program).result()
        function.algorithm = thought
        function.score = score
        function.evaluate_time = eval_time
        function.sample_time = sample_time
        function.operator = operator
        function._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        function._recipe_id = recipe.recipe_id
        function._eoh_generation_suggestion = suggestion
        return function if score is not None else None

    def __call__(self, population, recipe, selection_num, target_size):
        offspring = []
        operators = ("e1", "e2", "m1", "m2")
        next_operator = 0
        lock = threading.Lock()

        def sample_one(operator):
            refs = population.select_many(
                selection_num if operator in ("e1", "e2") else 1)
            return self._generate(refs, population, recipe, operator)

        while len(offspring) < target_size:
            batch_size = target_size - len(offspring)
            batch_operators = []
            for _ in range(batch_size):
                batch_operators.append(operators[next_operator % len(operators)])
                next_operator += 1
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(batch_size, self.num_samplers)) as executor:
                futures = [executor.submit(sample_one, operator)
                           for operator in batch_operators]
                for future in futures:
                    function = future.result()
                    if function is not None:
                        with lock:
                            offspring.append(function)
        return offspring

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
        """Generate one child using the original EoH operator prompt."""
        # UNCERTAIN: keep the same operator schedule as EoH for now.
        operator = getattr(self, "_next_operator", "e1")
        if operator == "e1":
            prompt = EoHPrompt.get_prompt_e1(refs, self.info, None)
        elif operator == "e2":
            prompt = EoHPrompt.get_prompt_e2(refs, self.info, None)
        elif operator == "m1":
            prompt = EoHPrompt.get_prompt_m1(refs[0], self.info, None)
        else:
            prompt = EoHPrompt.get_prompt_m2(refs[0], self.info, None)
        thought, function = self.sampler.get_thought_and_function(prompt)
        if thought is None or function is None:
            return None, operator
        program = TextFunctionProgramConverter.function_to_program(function, self.template_program)
        if program is None:
            return None, operator
        score, eval_time = self.executor.submit(
            self.evaluator.evaluate_program_record_time, program).result()
        function.algorithm = thought
        function.score = score
        function.evaluate_time = eval_time
        function.operator = operator
        function._recipe_id = recipe.recipe_id
        function._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        return (function if score is not None else None), operator

    def __call__(self, population, recipe, selection_num, target_size,
                 experiences=None):
        offspring, new_experiences = [], []
        parent_experiences = list(experiences or [])
        experience_lock = threading.RLock()
        manager = getattr(self.retrieve_experiences, "__self__", None)
        if manager is not None and hasattr(manager, "begin_node"):
            manager.begin_node(parent_experiences)
        operators = ("e1", "e2", "m1", "m2")
        next_operator = 0

        def sample_one(operator):
            refs = population.select_many(
                selection_num if operator in ("e1", "e2") else 1)
            retrieved = (self.retrieve_experiences(
                refs, operator, list(parent_experiences))
                if self.retrieve_experiences else [])
            fresh = (self.distill_experience(refs, operator)
                     if self.distill_experience else [])
            fresh = fresh if isinstance(fresh, list) else ([fresh] if fresh else [])
            guidance = self._format_experiences(retrieved + fresh)
            child, _ = self._generate_with_experience(
                refs, recipe, guidance, operator=operator)
            return child, refs, retrieved, fresh

        while len(offspring) < target_size:
            batch_size = target_size - len(offspring)
            batch_operators = []
            for _ in range(batch_size):
                batch_operators.append(operators[next_operator % len(operators)])
                next_operator += 1
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(batch_size, self.num_samplers)) as executor:
                results = list(executor.map(sample_one, batch_operators))
            with experience_lock:
                for child, refs, retrieved, fresh in results:
                    if child is None:
                        continue
                    offspring.append(child)
                    improved = self._objective_improved(child, refs)
                    retrieved_ids = {item.get("experience_id") for item in retrieved}
                    retained = []
                    for item in parent_experiences:
                        if item.get("experience_id") in retrieved_ids:
                            item["score"] = int(item.get("score", 0)) + (
                                1 if improved else -1)
                        if int(item.get("score", 0)) >= 0:
                            retained.append(item)
                    parent_experiences[:] = retained
                    new_experiences.extend(fresh)
        if isinstance(experiences, list):
            experiences[:] = parent_experiences
        if manager is not None and hasattr(manager, "end_node"):
            manager.end_node()
        return offspring, new_experiences

    def _generate_with_experience(self, refs, recipe, experience_text,
                                  operator=None):
        operator = operator or getattr(self, "_next_operator", "e1")
        if operator == "e1":
            prompt = EoHPrompt.get_prompt_e1(refs, self.info, experience_text or None)
        elif operator == "e2":
            prompt = EoHPrompt.get_prompt_e2(refs, self.info, experience_text or None)
        elif operator == "m1":
            prompt = EoHPrompt.get_prompt_m1(refs[0], self.info, experience_text or None)
        else:
            prompt = EoHPrompt.get_prompt_m2(refs[0], self.info, experience_text or None)
        if experience_text:
            prompt = prompt.replace(
                "These are some suggestions after reflecting on the given algorithms:",
                "These are successful and failed design experiences retrieved from previous attempts:")
        sample_start = time.time()
        thought, function = self.sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or function is None:
            return None, operator
        program = TextFunctionProgramConverter.function_to_program(function, self.template_program)
        if program is None:
            return None, operator
        score, eval_time = self.executor.submit(
            self.evaluator.evaluate_program_record_time, program).result()
        function.algorithm, function.score = thought, score
        function.evaluate_time, function.operator = eval_time, operator
        function.sample_time = sample_time
        function._recipe_id = recipe.recipe_id
        function._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        return (function if score is not None else None), operator

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
