from __future__ import annotations

import copy
import math
import random
import threading
import traceback

from ..mcts_ahd import MCTS_AHD
from ..mcts_ahd.mcts import MCTSNode
from ..mcts_ahd.prompt import MAEOHPrompt
from ..mcts_recipe.refineevo_experience import RefineEvoExperienceManager
from ..mcts_recipe.recipes import get_default_recipes


class MCTS_AHD_Recipe(MCTS_AHD):
    """MCTS-AHD variant with recipe selection before every expansion.

    The original MCTS-AHD implementation is untouched.  Tree selection,
    UCT, backpropagation and population survival remain inherited; only the
    prompt path is replaced with MCTS-AHD's own EoH-style prompt format plus
    a selected reflection recipe.
    """

    def __init__(self, *args, info=None, recipe_configs=None,
                 recipe_temperature=1.0, **kwargs):
        self._info = dict(info or {})
        if not self._info:
            raise ValueError("info is required for EoH-style recipe prompts")
        self._ensure_base_metadata(args, kwargs)
        super().__init__(*args, **kwargs)
        self._recipes = recipe_configs or get_default_recipes()
        self._temperature = float(recipe_temperature)
        if self._temperature <= 0:
            raise ValueError("recipe_temperature must be positive")
        self._recipe_gain = {name: 0.0 for name in self._recipes}
        self._recipe_count = {name: 0 for name in self._recipes}
        self._recipe_lock = threading.RLock()
        self._refineevo_manager = RefineEvoExperienceManager(
            reflector_llm=self._sampler.llm,
            embedding_model=None,
            top_k=max(int(cfg.get("experience_top_k", 3))
                      for cfg in self._recipes.values()),
        )

    def _ensure_base_metadata(self, args, kwargs):
        evaluation = kwargs.get("evaluation")
        if evaluation is None and len(args) >= 2:
            evaluation = args[1]
        if evaluation is None:
            return
        if not getattr(evaluation, "template_program", None):
            method_name = self._info["method_name"]
            signature = self._info["method_signature"]
            evaluation.template_program = f"def {method_name}({signature}):\n    pass\n"
        if not getattr(evaluation, "task_description", None):
            evaluation.task_description = self._info["task_description"]

    def _choose_recipe(self):
        with self._recipe_lock:
            names = list(self._recipes)
            values = [self._recipe_gain[name] / self._temperature for name in names]
        shift = max(values)
        weights = [math.exp(value - shift) for value in values]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        index = random.choices(range(len(names)), weights=probabilities, k=1)[0]
        return names[index], probabilities[index]

    @staticmethod
    def _lineage_parents(refs):
        return [list(getattr(ref, "_recipe_parent_functions", ())) for ref in refs]

    @staticmethod
    def _refineevo_operator(operator):
        return "e1" if operator == "s1" else operator

    @staticmethod
    def _format_experiences(experiences):
        if not experiences:
            return None
        lines = []
        for index, item in enumerate(experiences, 1):
            status = item.get("is_success")
            if status is True:
                label = "successful"
            elif status is False:
                label = "failed"
            else:
                label = "reference"
            recommendations = item.get("recommendations") or []
            if isinstance(recommendations, str):
                recommendations = [recommendations]
            lines.append(
                f"Experience {index} ({label}):\n"
                f"Summary: {item.get('summary', '')}\n"
                f"Recommendations: {'; '.join(str(x) for x in recommendations)}\n"
                f"Applicable when: {item.get('applicable_when', '')}"
            )
        return "\n\n".join(lines)

    def _reflection(self, refs, recipe_id):
        config = self._recipes[recipe_id]
        prompt = MAEOHPrompt.get_prompt_reflection(
            refs=list(refs), parents=self._lineage_parents(refs), info=self._info,
            parent_info_flag=config.get("parent_info", False),
            best_worst_flag=config.get("best_worst", False),
            fitness_flag=config.get("fitness", 0),
            avg_fitness_flag=config.get("avg_fitness", False),
            check_reflection_flag=config.get("check_guidance", False),
            population=self._population,
            use_long_term_reflection=False,
            identical_parent_children_flag=False,
            shared_parent_children_flag=False,
            population_comparison=config.get("population_comparison"),
            comparison_flag=config.get("comparison", False),
            attribution_flag=config.get("attribution", False),
            summarization_flag=config.get("summarization", False),
            attribution_task=config.get("attribution_type", "good"),
            summarization_task=config.get("summarization_type", "guidance"),
        )
        result = self._sampler.llm.draw_sample(prompt)
        return result.strip() if isinstance(result, str) else None

    def _refineevo_reflection(self, refs, operator, cur_node, recipe_id):
        config = self._recipes[recipe_id]
        parent_experiences = copy.deepcopy(
            getattr(cur_node, "_refineevo_experiences", []) or []
        )
        self._refineevo_manager.top_k = int(config.get("experience_top_k", 3))
        manager_operator = self._refineevo_operator(operator)
        self._refineevo_manager.begin_node(parent_experiences)
        try:
            retrieved = self._refineevo_manager.retrieve(
                refs, manager_operator, parent_experiences)
            fresh = self._refineevo_manager.distill(refs, manager_operator)
        finally:
            self._refineevo_manager.end_node()
        return (self._format_experiences(retrieved + fresh),
                parent_experiences, retrieved, fresh)

    def _eoh_operator_prompt(self, operator, refs, suggestion,
                             suggestion_title=None):
        if operator == "e1":
            return MAEOHPrompt.get_prompt_e1(
                refs, self._info, suggestion, suggestion_title)
        if operator == "s1":
            return MAEOHPrompt.get_prompt_s1(
                refs, self._info, suggestion, suggestion_title)
        if operator == "e2":
            return MAEOHPrompt.get_prompt_e2(
                refs, self._info, suggestion, suggestion_title)
        if operator == "m1":
            return MAEOHPrompt.get_prompt_m1(
                refs[0], self._info, suggestion, suggestion_title)
        if operator == "m2":
            return MAEOHPrompt.get_prompt_m2(
                refs[0], self._info, suggestion, suggestion_title)
        raise ValueError(f"Invalid MCTS-AHD operator: {operator}")

    def _register_expanded_function(self, mcts, node_set, cur_node, func,
                                    operator, refs, recipe_id, probability,
                                    parent_experiences=None,
                                    retrieved_experiences=None,
                                    fresh_experiences=None):
        if func is False or func is None or func.score is None:
            return node_set
        if self.check_duplicate(node_set, str(func)):
            return node_set
        func.recipe_id = recipe_id
        func.recipe_probability = probability
        func.operator = operator
        func._eoh_generation_suggestion = getattr(func, "_recipe_suggestion", None)
        baseline = max((float(ref.score) for ref in refs if ref.score is not None),
                       default=None)
        gain = max(0.0, float(func.score) - baseline) if baseline is not None else 0.0
        with self._recipe_lock:
            self._recipe_gain[recipe_id] += gain
            self._recipe_count[recipe_id] += 1
        func._recipe_parent_functions = tuple(copy.deepcopy(refs))
        self._population.register_function(func)
        node = MCTSNode(func.algorithm, str(func), -1 * func.score,
                        individual=func, parent=cur_node, depth=cur_node.depth + 1,
                        visit=1, Q=func.score, raw_info=func)
        experiences = copy.deepcopy(
            parent_experiences if parent_experiences is not None
            else getattr(cur_node, "_refineevo_experiences", []) or []
        )
        retrieved_ids = {
            item.get("experience_id") for item in (retrieved_experiences or [])
            if item.get("experience_id")
        }
        if retrieved_ids:
            delta = 1 if gain > 0 else -1
            for item in experiences:
                if item.get("experience_id") in retrieved_ids:
                    item["score"] = int(item.get("score", 0)) + delta
        experiences.extend(copy.deepcopy(fresh_experiences or []))
        node._refineevo_experiences = experiences
        if operator == "e1":
            node.subtree.append(node)
        cur_node.add_child(node)
        mcts.backpropagate(node)
        node_set.append(node)
        return node_set

    def expand(self, mcts, node_set, cur_node, option):
        """Expand one MCTS action using a recipe and EoH operator prompt."""
        if option == "e1":
            refs = [copy.deepcopy(child.subtree[0].individual)
                    for child in mcts.root.children if child.subtree]
            refs = refs or [copy.deepcopy(cur_node.individual)]
        elif option == "e2":
            other = self._population.selection()
            refs = [other, copy.deepcopy(cur_node.individual)]
        else:
            refs = [copy.deepcopy(cur_node.individual)]

        recipe_id, probability = self._choose_recipe()
        try:
            config = self._recipes[recipe_id]
            parent_experiences = None
            retrieved_experiences = None
            fresh_experiences = None
            suggestion_title = None
            if config.get("refineevo_experience", False):
                suggestion, parent_experiences, retrieved_experiences, fresh_experiences = (
                    self._refineevo_reflection(refs, option, cur_node, recipe_id)
                )
                suggestion_title = "These are successful and failed design experiences retrieved from previous attempts:"
            else:
                suggestion = self._reflection(refs, recipe_id)
            prompt = self._eoh_operator_prompt(
                option, refs, suggestion, suggestion_title)
            func = self._sample_evaluate_register(prompt, func_only=True)
            if func is not False and func is not None:
                func._recipe_suggestion = suggestion
            return self._register_expanded_function(
                mcts, node_set, cur_node, func, option, refs,
                recipe_id, probability, parent_experiences,
                retrieved_experiences, fresh_experiences)
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return node_set

    def _iteratively_init_population_root(self):
        while len(self._population.population) < self._init_pop_size:
            try:
                prompt = MAEOHPrompt.get_prompt_e1(
                    self._population.population, self._info)
                self._sample_evaluate_register(prompt)
                self._population.survival()
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(
                        f'Note: During initialization, MCTS-AHD gets '
                        f'{len(self._population) + len(self._population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _init_one_solution(self):
        while len(self._population.next_gen_pop) == 0:
            try:
                prompt = MAEOHPrompt.get_prompt_i1(self._info)
                self._sample_evaluate_register(prompt)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue
