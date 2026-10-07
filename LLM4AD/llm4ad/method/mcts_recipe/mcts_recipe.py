from __future__ import annotations

import copy
import json
import os
import base64
import concurrent.futures
import math
import pickle
import random
import threading
from dataclasses import dataclass

from .budget import SampleBudgetExhausted
from .adaptive_router import ContextualThompsonRouter
from .mcts import RecipeMCTS, RecipeNode
from .population import RecipePopulation
from .persistence import RecipeStore
from .recipes import validate_recipes

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is optional for the tree itself.
    np = None


@dataclass
class RecipeConfig:
    """A recipe is a complete reflection configuration.

    The fields are intentionally open-ended until the final recipe set is fixed.
    """

    recipe_id: str
    values: dict


class MCTSRecipe:
    """MCTS whose actions are reflection recipes and states are populations.

    ``expand_fn`` is injected so the EoH-specific prompt/operator pipeline can
    evolve independently of this tree implementation. It must have signature
    ``expand_fn(population, recipe_config, selection_num, target_size)`` and
    return a list of feasible Function objects. The callback owns the EoH
    reference selection, reflection prompt, operator scheduling, evaluation,
    and retry-until-a-generation-is-complete logic.
    """

    def __init__(self, recipes, pop_size=10, selection_num=2,
                 max_depth=50, exploration_constant=0.1,
                 depth_balance_weight=0.0,
                 node_batch_size=100, store_dir="recipe_mcts",
                 expand_fn=None, seed=None, max_sample_count=None,
                 elite_pool_size=0, adaptive_config=None):
        recipes = validate_recipes(recipes)
        self.recipes = {
            k: (v if isinstance(v, RecipeConfig) else RecipeConfig(k, dict(v)))
            for k, v in recipes.items()
        }
        self.pop_size = int(pop_size)
        self.selection_num = int(selection_num)
        self.max_depth = int(max_depth)
        self.expand_fn = expand_fn
        self.elite_pool_size = int(elite_pool_size or 0)
        if self.elite_pool_size < 0:
            raise ValueError("elite_pool_size must be >= 0")
        self.use_elite_pool = self.elite_pool_size > 0
        self.adaptive_config = dict(adaptive_config or {})
        self.adaptive_enabled = bool(
            self.adaptive_config.get("enabled", False))
        self.adaptive_recipe_count = int(
            self.adaptive_config.get("recipes_per_expansion", 3))
        if self.adaptive_recipe_count < 1:
            raise ValueError("recipes_per_expansion must be >= 1")
        self.adaptive_reward_best_weight = float(
            self.adaptive_config.get("reward_best_weight", 0.8))
        self.adaptive_reward_mean_weight = float(
            self.adaptive_config.get("reward_mean_weight", 0.2))
        self.adaptive_router = None
        if self.adaptive_enabled:
            self.adaptive_router = ContextualThompsonRouter(
                self.recipes.keys(),
                prior_precision=float(
                    self.adaptive_config.get("prior_precision", 1.0)),
                posterior_scale=float(
                    self.adaptive_config.get("posterior_scale", 0.5)),
            )
        # UNCERTAIN: the exact recipe definitions and EoH integration callback
        # are deliberately supplied by the task runner, not hard-coded here.
        self.depth_balance_weight = float(depth_balance_weight)
        self.tree = RecipeMCTS(
            self.recipes, exploration_constant, max_depth,
            depth_balance_weight=self.depth_balance_weight)
        self.store = RecipeStore(store_dir, node_batch_size)
        # All recipe-MCTS variants share the disk-backed tree index. The
        # selected node's population snapshot is loaded only when expanded.
        self._disk_tree_mode = True
        self._tree_checkpoint_pending = False
        # Compact, non-algorithm tree metadata generated after the last node
        # snapshot flush. It is committed only after the corresponding node
        # records are durable, preventing a checkpoint from pointing at a
        # node that still exists only in RecipeStore.buffer.
        self._tree_root_id = None
        self._pending_tree_updates = {}
        if max_sample_count is not None:
            max_sample_count = int(max_sample_count)
            if max_sample_count < 1:
                raise ValueError("max_sample_count must be >= 1")
        self.max_sample_count = max_sample_count
        self._sample_limit_logged = False
        self._next_population_node_id = 1
        self._next_algorithm_id = 1
        self._checkpoint_index = max((
            int(entry.name[len("checkpoint_"):-5])
            for entry in os.scandir(self.store.directory)
            if entry.name.startswith("checkpoint_") and entry.name.endswith(".json")
            and entry.name[len("checkpoint_"):-5].isdigit()
        ), default=0)
        self._tree_lock = threading.RLock()
        self._elite_pool = self._load_elite_pool() if self.use_elite_pool else []
        self._best_samples = self.store.load_best_samples()
        self._convergence_records = self.store.load_convergence_records()
        self._convergence_algorithm_ids = {
            int(record["algorithm_id"])
            for record in self._convergence_records
            if isinstance(record, dict)
            and record.get("algorithm_id") is not None
        }
        self._convergence_dirty = False
        self._last_convergence_write_count = len(self._convergence_records)
        # Keep the append-only improvement history used by EoH.  The current
        # global best is the last/highest record, while every strict
        # improvement remains available for convergence analysis.
        valid_best = [record for record in self._best_samples
                      if isinstance(record, dict)
                      and record.get("score") is not None]
        if valid_best:
            best_record = max(valid_best, key=lambda item: float(item["score"]))
            self._best_samples = list(self._best_samples)
            self._best_score = float(best_record["score"])
        else:
            self._best_samples = []
            self._best_score = float("-inf")
        self._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._total_sample_count = 0
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)

    def initialize(self, individuals, initial_samples_recorded=False,
                   initial_token_usage=None):
        population = RecipePopulation(copy.deepcopy(individuals), self.pop_size)
        if not population.individuals:
            raise ValueError("initial population cannot be empty")
        self._assign_algorithm_ids(population.individuals)
        best, mean, std = self._population_stats(population)
        root_id = self._next_population_node_id
        self._next_population_node_id += 1
        root = RecipeNode(
            root_id,
            best,
            depth=0,
            token_usage=initial_token_usage,
            stagnation_iteration=0,
            best_score=best,
            score_mean=mean,
            score_std=std,
        )
        self.tree.root = root
        if not initial_samples_recorded:
            self._record_evaluated_candidates(population.individuals, root)
        self._update_elite_pool(population.individuals)
        self._write_node(root, population, None, [], initial_token_usage)
        self.store.flush()
        self._set_pending_tree_state(root_id, [{
            "population_node_id": root_id,
            "parent_population_node_id": None,
            "depth": 0,
            "Q": float(root.Q),
            "visits": int(root.visits),
            "children_population_node_ids": [],
            "expanded_recipe_ids": [],
            "incoming_recipe_id": None,
            "token_usage": dict(initial_token_usage or {}),
            "stagnation_iteration": 0,
            "best_score": best,
            "score_mean": mean,
            "score_std": std,
        }])
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self.tree.root = None
        return root

    def _assign_algorithm_ids(self, individuals):
        with self._tree_lock:
            for item in individuals:
                if getattr(item, "_recipe_algorithm_id", None) is None:
                    item._recipe_algorithm_id = self._next_algorithm_id
                    self._next_algorithm_id += 1

    def _attach_parent_functions(self, functions):
        from .resume import function_from_record

        parent_ids = {
            parent_id
            for function in functions
            for parent_id in getattr(function, "_eoh_parent_ids", ())
            if parent_id is not None
        }
        parent_records = self.store.load_algorithm_records(parent_ids)
        parent_functions = {
            parent_id: function_from_record(parent_record)
            for parent_id, parent_record in parent_records.items()
        }
        for function in functions:
            function._recipe_parent_functions = tuple(
                parent_functions[parent_id]
                for parent_id in getattr(function, "_eoh_parent_ids", ())
                if parent_id in parent_functions
            )

    def _load_elite_pool(self, algorithm_ids=None):
        from .resume import function_from_record

        records = self.store.load_elite_pool_records()
        if not records and algorithm_ids:
            record_by_id = self.store.load_algorithm_records(algorithm_ids)
            records = [record_by_id[int(algorithm_id)]
                       for algorithm_id in algorithm_ids
                       if int(algorithm_id) in record_by_id]
        functions = []
        for record in records:
            try:
                functions.append(function_from_record(record))
            except Exception as exc:
                print(f"Skipping invalid elite-pool record: {exc}", flush=True)
        elite = RecipePopulation.best_unique(functions, self.elite_pool_size)
        self._attach_parent_functions(elite)
        return elite

    @staticmethod
    def _elite_signature(individuals):
        return tuple(
            (getattr(item, "_recipe_algorithm_id", None),
             float(getattr(item, "score", float("-inf"))),
             str(item))
            for item in individuals
        )

    def _update_elite_pool(self, individuals):
        if not self.use_elite_pool:
            return False
        old_signature = self._elite_signature(self._elite_pool)
        candidates = list(self._elite_pool) + copy.deepcopy(list(individuals))
        self._elite_pool = RecipePopulation.best_unique(
            candidates, self.elite_pool_size)
        self._attach_parent_functions(self._elite_pool)
        if self._elite_signature(self._elite_pool) == old_signature:
            return False
        self.store.write_elite_pool(self._elite_pool, self.elite_pool_size)
        return True

    def _reference_population(self, population, elite_snapshot):
        if not self.use_elite_pool or not elite_snapshot:
            return population
        return population.with_extra_reference_individuals(elite_snapshot)

    def reserve_sample_order(self):
        """Reserve one EoH-style sampling attempt before generation starts.

        The order counts calls to i1/e1/e2/m1/m2, including invalid code or
        failed evaluations.  It is intentionally independent of feasible
        algorithm IDs.
        """
        with self._tree_lock:
            if self.sample_budget_exhausted():
                self._log_sample_limit_once_locked()
                raise SampleBudgetExhausted(
                    f"sample limit reached: {self._total_sample_count}/"
                    f"{self.max_sample_count}")
            self._total_sample_count += 1
            return self._total_sample_count

    def sample_budget_exhausted(self):
        return (self.max_sample_count is not None
                and self._total_sample_count >= self.max_sample_count)

    def _log_sample_limit_once_locked(self):
        if not self._sample_limit_logged:
            print(
                f"Sample limit reached "
                f"({self._total_sample_count}/{self.max_sample_count}); "
                "no new samples will be started. Waiting for running "
                "evaluations to finish.",
                flush=True,
            )
            self._sample_limit_logged = True

    def record_evaluated_sample(self, individual, population_node_id, depth):
        """Register one completed evaluation immediately and atomically.

        This is deliberately called by the thread receiving an evaluation
        future, rather than after a recipe batch or node is persisted.
        """
        with self._tree_lock:
            self._assign_algorithm_ids([individual])
            context = type("SampleContext", (), {
                "population_node_id": int(population_node_id),
                "depth": int(depth),
            })()
            self._record_evaluated_candidates([individual], context)

    def _record_evaluated_candidates(self, individuals, population_node):
        """Persist global-best and best-so-far records for evaluated algorithms."""
        with self._tree_lock:
            any_appended = False
            for individual in individuals:
                self._record_best_candidates([individual], population_node)
                any_appended = self._append_convergence_record(
                    individual, population_node) or any_appended
            if not any_appended:
                return
            self._convergence_dirty = True
            self._write_convergence_outputs(plot=False)

    def _append_convergence_record(self, individual, population_node):
        algorithm_id = getattr(individual, "_recipe_algorithm_id", None)
        score = getattr(individual, "score", None)
        if algorithm_id is None or score is None:
            return False
        try:
            algorithm_id = int(algorithm_id)
            score = float(score)
        except (TypeError, ValueError):
            return False
        if algorithm_id in self._convergence_algorithm_ids:
            return False
        self._convergence_algorithm_ids.add(algorithm_id)
        self._convergence_records.append({
            "algorithm_id": algorithm_id,
            "sample_order": getattr(individual, "_recipe_sample_order", None),
            "population_node_id": population_node.population_node_id,
            "depth": population_node.depth,
            "recipe_id": getattr(individual, "_recipe_id", None),
            "operator": getattr(individual, "operator", None),
            "score": score,
            "best_score_so_far": float(self._best_score),
        })
        return True

    def _write_convergence_outputs(self, plot=False):
        if not self._convergence_records:
            return
        if self._convergence_dirty:
            self.store.write_convergence_records(self._convergence_records)
            self._last_convergence_write_count = len(self._convergence_records)
            self._convergence_dirty = False
        if plot:
            try:
                self.store.write_convergence_plot(self._convergence_records)
            except Exception as exc:
                print(f"WARNING: failed to write convergence plot: {exc}",
                      flush=True)

    def _record_best_candidates(self, individuals, population_node):
        """Append and persist each strict global-best improvement.

        This method is called only after an individual has been evaluated.  A
        candidate is compared with the process-wide ``_best_score``; when it
        improves that value, the record is appended and ``sample_best.json``
        is written immediately.  It is intentionally independent of node
        population batching.
        """
        with self._tree_lock:
            for individual in individuals:
                score = getattr(individual, "score", None)
                if score is None or score <= self._best_score:
                    continue
                code = (individual.to_code_without_docstring()
                        if hasattr(individual, "to_code_without_docstring")
                        else str(individual))
                self._best_score = float(score)
                best_record = {
                    "sample_order": getattr(
                        individual, "_recipe_sample_order", None),
                    "algorithm_id": getattr(individual, "_recipe_algorithm_id", None),
                    "population_node_id": population_node.population_node_id,
                    "depth": population_node.depth,
                    "recipe_id": getattr(individual, "_recipe_id", None),
                    "operator": getattr(individual, "operator", None),
                    "parent_algorithm_ids": list(
                        getattr(individual, "_eoh_parent_ids", ())),
                    "algorithm": getattr(individual, "algorithm", ""),
                    "function": code,
                    "program": code,
                    "score": score,
                    "evaluate_time": getattr(individual, "evaluate_time", None),
                    "sample_time": getattr(individual, "sample_time", None),
                    "token_usage": getattr(individual, "_recipe_token_usage", None),
                }
                self._best_samples.append(best_record)
                self.store.write_best_samples(self._best_samples)

    def _write_node(self, node, population, recipe_id, experiences,
                    token_usage=None):
        if token_usage is None:
            token_usage = getattr(node, "token_usage", None)
        self._accumulate_token_usage(token_usage)
        flushed = self.store.add(
            node.population_node_id, node.parent_population_node_id,
            node.depth, recipe_id, population.generation, population,
            experiences, token_usage=token_usage)
        if flushed is not None:
            if self._disk_tree_mode:
                self._tree_checkpoint_pending = True
            else:
                self._checkpoint_index += 1
                self._write_checkpoint()

    def _accumulate_token_usage(self, usage):
        """Add one node/initialization aggregate to the checkpoint total."""
        if not isinstance(usage, dict):
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        self._cumulative_token_usage["prompt_tokens"] += prompt
        self._cumulative_token_usage["completion_tokens"] += completion
        self._cumulative_token_usage["total_tokens"] += int(
            usage.get("total_tokens", prompt + completion) or prompt + completion)

    def _load_node_state(self, node_id):
        from .resume import function_from_record

        record = self.store.load_node(node_id)
        functions = [function_from_record(item)
                     for item in record.get("algorithms", [])]
        self._attach_parent_functions(functions)
        population = RecipePopulation(functions, self.pop_size,
                                      record.get("generation", 0))
        return population, list(record.get("experiences", []))

    @staticmethod
    def _population_stats(population):
        scores = np.asarray([
            float(item.score) for item in population.individuals
            if getattr(item, "score", None) is not None
        ], dtype=float)
        if scores.size == 0:
            raise ValueError("population has no scored individuals")
        return float(np.max(scores)), float(np.mean(scores)), float(np.std(scores))

    def _adaptive_state(self, node, population):
        best, mean, std = self._population_stats(population)
        return [
            float(node.depth),
            float(getattr(node, "stagnation_iteration", 0)),
            best,
            mean,
            std,
        ]

    def _adaptive_reward(self, parent_population, child_population):
        parent_best, parent_mean, _ = self._population_stats(parent_population)
        child_best, child_mean, _ = self._population_stats(child_population)
        best_gain = child_best - parent_best
        mean_gain = child_mean - parent_mean
        return (
            self.adaptive_reward_best_weight * best_gain
            + self.adaptive_reward_mean_weight * mean_gain
        )

    def expand_node(self, node, population, experiences=None):
        """Expand pending recipes from one shared parent snapshot."""
        if self.expand_fn is None:
            raise NotImplementedError(
                "Provide expand_fn implementing EoH reflection + operator scheduling.")
        if self.sample_budget_exhausted():
            with self._tree_lock:
                self._log_sample_limit_once_locked()
            return []
        pending_recipes = [
            (recipe_id, recipe)
            for recipe_id, recipe in self.recipes.items()
            if recipe_id not in node.expanded_recipe_ids
        ]
        if not pending_recipes:
            raise RuntimeError(
                f"Population node {node.population_node_id} is already fully expanded")
        adaptive_state = None
        sampled_recipe_values = {}
        if self.adaptive_enabled:
            adaptive_state = self._adaptive_state(node, population)
            selected_ids, sampled_recipe_values = (
                self.adaptive_router.select_top_k(
                    adaptive_state,
                    [recipe_id for recipe_id, _ in pending_recipes],
                    k=self.adaptive_recipe_count,
                )
            )
            selected_set = set(selected_ids)
            pending_recipes = [
                (recipe_id, recipe)
                for recipe_id, recipe in pending_recipes
                if recipe_id in selected_set
            ]
            print(
                f"Adaptive recipe selection node={node.population_node_id} "
                f"state={adaptive_state} selected="
                f"{[recipe_id for recipe_id, _ in pending_recipes]}",
                flush=True,
            )
        children = []

        with self._tree_lock:
            child_ids = {
                recipe_id: self._next_population_node_id + index
                for index, (recipe_id, _) in enumerate(pending_recipes)
            }
            self._next_population_node_id += len(pending_recipes)
            elite_snapshot = (copy.deepcopy(self._elite_pool)
                              if self.use_elite_pool else [])

        def expand_recipe(recipe_id, recipe):
            with self._tree_lock:
                if self.sample_budget_exhausted():
                    self._log_sample_limit_once_locked()
                    return
            branch_experiences = copy.deepcopy(list(experiences or []))
            branch_population = population.clone()
            reference_population = self._reference_population(
                branch_population, elite_snapshot)
            expander = (self.expand_fn[recipe_id]
                        if isinstance(self.expand_fn, dict) else self.expand_fn)
            attempted_samples = 0

            def sample_order_callback():
                nonlocal attempted_samples
                order = self.reserve_sample_order()
                attempted_samples += 1
                return order

            sample_callback = lambda individual: self.record_evaluated_sample(
                individual, child_ids[recipe_id], node.depth + 1)
            if recipe.values.get("refineevo_experience", False):
                result = expander(
                    reference_population, recipe, self.selection_num,
                    self.pop_size, experiences=branch_experiences,
                    on_evaluated=sample_callback,
                    reserve_sample_order=sample_order_callback)
            else:
                result = expander(
                    reference_population, recipe, self.selection_num, self.pop_size,
                    on_evaluated=sample_callback,
                    reserve_sample_order=sample_order_callback)
            if isinstance(result, tuple):
                offspring, new_experiences = result
            else:
                offspring, new_experiences = result, []
            node_token_usage = getattr(expander, "last_node_token_usage", None)
            if not offspring and attempted_samples == 0 \
                    and self.sample_budget_exhausted():
                return
            with self._tree_lock:
                if offspring:
                    self._assign_algorithm_ids(offspring)
                    child_population = branch_population.survival(offspring)
                else:
                    child_population = branch_population.inherited_next_generation()
                child_id = child_ids[recipe_id]
                child_best, child_mean, child_std = self._population_stats(
                    child_population)
                stagnation_iteration = (
                    0 if child_best > node.best_score
                    else node.stagnation_iteration + 1
                )
                child = RecipeNode(
                    child_id,
                    child_best,
                    depth=node.depth + 1, parent=node,
                    incoming_recipe_id=recipe_id,
                    token_usage=node_token_usage,
                    stagnation_iteration=stagnation_iteration,
                    best_score=child_best,
                    score_mean=child_mean,
                    score_std=child_std,
                )
                node.add_child(child)
                # One completed recipe edge is one visit to its new child and
                # one additional visit propagated through every ancestor.
                self.tree.backpropagate(child)
                branch_experiences.extend(new_experiences)
                self._update_elite_pool(child_population.individuals)
                if self.adaptive_enabled:
                    reward = self._adaptive_reward(
                        population, child_population)
                    self.adaptive_router.update(
                        recipe_id, adaptive_state, reward)
                self._write_node(child, child_population, recipe_id,
                                 branch_experiences, node_token_usage)
                elite_text = (f" elite_pool={len(self._elite_pool)}/"
                              f"{self.elite_pool_size}"
                              if self.use_elite_pool else "")
                print(
                    f"Generated population node {child_id} "
                    f"recipe={recipe_id} depth={child.depth} "
                    f"tokens={node_token_usage}{elite_text}",
                    flush=True,
                )
                children.append(child)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(pending_recipes)) as executor:
            futures = [executor.submit(expand_recipe, recipe_id, recipe)
                       for recipe_id, recipe in pending_recipes]
            for future in futures:
                future.result()
        return children

    def run(self, initial_individuals=None, checkpoint=None,
            initial_samples_recorded=False, initial_token_usage=None):
        if checkpoint is not None:
            self.restore(checkpoint)
        else:
            if initial_individuals is None:
                raise ValueError("initial_individuals is required for a new run")
            if self.store.has_population_nodes():
                raise FileExistsError(
                    f"{self.store.directory} already contains population nodes; "
                    "use a new LLM4AD_LOG_DIR or set LLM4AD_CHECKPOINT")
            self.initialize(initial_individuals, initial_samples_recorded,
                            initial_token_usage)
        while True:
            if self.sample_budget_exhausted():
                with self._tree_lock:
                    self._log_sample_limit_once_locked()
                break
            node = (self._select_disk_tree_node()
                    if self._disk_tree_mode else self.tree.select())
            if node is None:
                break
            if node.depth >= self.max_depth:
                break
            population, experiences = self._load_node_state(node.population_node_id)
            children = self.expand_node(node, population, experiences)
            if self._disk_tree_mode:
                self._write_disk_tree_transition(node, children)
                if self._tree_checkpoint_pending:
                    self.store.flush()
                    self._commit_pending_tree_state()
                    self._checkpoint_index += 1
                    self._write_checkpoint()
                    self._tree_checkpoint_pending = False
                has_children = bool(children)
                del children, population, node
                self.tree.root = None
                if self.sample_budget_exhausted():
                    with self._tree_lock:
                        self._log_sample_limit_once_locked()
                    break
                if not has_children:
                    break
                continue
            if self.sample_budget_exhausted():
                with self._tree_lock:
                    self._log_sample_limit_once_locked()
                break
            if not children or any(child.depth >= self.max_depth for child in children):
                break
        self.store.flush()
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self._write_convergence_outputs(plot=True)
        return self.tree.root

    def _disk_tree_records(self):
        payload = self._disk_tree_payload()
        records = {}
        for item in payload.get("nodes", []):
            try:
                records[int(item["population_node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        return payload.get("root_id"), records

    def _disk_tree_payload(self):
        payload = self.store.load_tree_state()
        root_id = (self._tree_root_id
                   if self._tree_root_id is not None
                   else payload.get("root_id"))
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["population_node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        for node_id, item in self._pending_tree_updates.items():
            by_id[int(node_id)] = dict(item)
        return {"root_id": root_id, "nodes": list(by_id.values())}

    def _set_pending_tree_state(self, root_id, records, changed_ids=None):
        self._tree_root_id = int(root_id) if root_id is not None else None
        if isinstance(records, dict):
            for node_id in changed_ids or ():
                if node_id in records:
                    self._pending_tree_updates[int(node_id)] = dict(
                        records[node_id])
        else:
            for item in records:
                if item.get("population_node_id") is not None:
                    self._pending_tree_updates[
                        int(item["population_node_id"])] = dict(item)

    def _commit_pending_tree_state(self):
        if not self._pending_tree_updates:
            return
        payload = self.store.load_tree_state()
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["population_node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        for node_id, item in self._pending_tree_updates.items():
            by_id[int(node_id)] = dict(item)
        self.store.write_tree_state(
            self._tree_root_id if self._tree_root_id is not None
            else payload.get("root_id"),
            list(by_id.values()),
        )
        self._pending_tree_updates = {}

    def _select_disk_tree_node(self):
        """Select one node using metadata loaded from disk for this call only."""
        root_id, records = self._disk_tree_records()
        if root_id is None or root_id not in records:
            raise KeyError("disk-backed MCTS tree has no root")

        children = {}
        for item in records.values():
            parent_id = item.get("parent_population_node_id")
            if parent_id is not None:
                children.setdefault(int(parent_id), []).append(item)

        recipe_count = len(self.recipes)
        candidates = []
        for item in records.values():
            if int(item.get("depth", 0)) >= self.max_depth:
                continue
            expanded = set(item.get("expanded_recipe_ids", []))
            if len(expanded) < recipe_count:
                candidates.append(item)
        if not candidates:
            return None

        def uct(item, siblings):
            q_values = [float(x.get("Q", 0.0)) for x in siblings]
            q_min = min(q_values, default=float(item.get("Q", 0.0)))
            q_max = max(q_values, default=float(item.get("Q", 0.0)))
            q = float(item.get("Q", 0.0))
            exploit = ((q - q_min) / (q_max - q_min)
                       if q_max > q_min else 0.0)
            parent_id = item.get("parent_population_node_id")
            parent_visits = float(
                records.get(int(parent_id), {}).get("visits", 1.0)
                if parent_id is not None else 1.0
            )
            visits = max(float(item.get("visits", 1.0)), 1.0)
            return exploit + self.tree.exploration_constant * math.sqrt(
                math.log(parent_visits + 1.0) / visits)

        if self.depth_balance_weight == 0.0:
            current = records[root_id]
            while children.get(int(current["population_node_id"]), []):
                current_children = children[int(current["population_node_id"])]
                if len(set(current.get("expanded_recipe_ids", []))) < recipe_count:
                    break
                current = max(
                    current_children,
                    key=lambda item: uct(item, current_children),
                )
            selected = current
            if (int(selected.get("depth", 0)) < self.max_depth
                    and len(set(selected.get("expanded_recipe_ids", []))) < recipe_count):
                pass
            else:
                selected = max(candidates, key=lambda item: float(item.get("Q", 0.0)))
        else:
            depth_counts = {}
            for item in records.values():
                depth = int(item.get("depth", 0))
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
            max_count = max(depth_counts.values(), default=1)
            deepest = max((int(item.get("depth", 0)) for item in candidates), default=0)
            selected = max(
                candidates,
                key=lambda item: (
                    uct(item, children.get(int(item["parent_population_node_id"]), []))
                    if item.get("parent_population_node_id") is not None else float("inf")
                ) + self.depth_balance_weight * (
                    (max_count - depth_counts.get(int(item.get("depth", 0)), 0))
                    / (max_count + 1.0)
                    + 0.25 * int(item.get("depth", 0)) / max(deepest, 1)
                ),
            )

        from .mcts import RecipeNode
        node = RecipeNode(
            int(selected["population_node_id"]),
            float(selected.get("Q", selected.get("best_score", 0.0))),
            depth=int(selected.get("depth", 0)),
            incoming_recipe_id=selected.get("incoming_recipe_id"),
            visits=int(selected.get("visits", 1)),
            token_usage=selected.get("token_usage"),
            stagnation_iteration=int(selected.get("stagnation_iteration", 0)),
            best_score=selected.get("best_score"),
            score_mean=selected.get("score_mean"),
            score_std=selected.get("score_std"),
        )
        node.parent_population_node_id = selected.get(
            "parent_population_node_id")
        node.expanded_recipe_ids = set(selected.get("expanded_recipe_ids", []))
        return node

    def _write_disk_tree_transition(self, parent_node, children):
        """Update only the on-disk tree index after a node expansion."""
        root_id, records = self._disk_tree_records()
        if root_id is None:
            root_id = parent_node.population_node_id
        parent_id = int(parent_node.population_node_id)
        parent = records.setdefault(parent_id, {
            "population_node_id": parent_id,
            "parent_population_node_id": None,
            "depth": int(parent_node.depth),
            "Q": float(parent_node.Q),
            "visits": int(parent_node.visits),
            "expanded_recipe_ids": [],
            "incoming_recipe_id": parent_node.incoming_recipe_id,
            "token_usage": getattr(parent_node, "token_usage", None),
            "stagnation_iteration": int(parent_node.stagnation_iteration),
            "best_score": float(parent_node.best_score),
            "score_mean": parent_node.score_mean,
            "score_std": parent_node.score_std,
        })
        parent["expanded_recipe_ids"] = sorted(
            set(parent.get("expanded_recipe_ids", [])) |
            {child.incoming_recipe_id for child in children
             if child.incoming_recipe_id is not None}
        )
        parent["children_population_node_ids"] = sorted(
            {int(item.get("population_node_id")) for item in records.values()
             if item.get("parent_population_node_id") == parent_id}
            | {int(child.population_node_id) for child in children}
        )
        for child in children:
            child_id = int(child.population_node_id)
            records[child_id] = {
                "population_node_id": child_id,
                "parent_population_node_id": parent_id,
                "depth": int(child.depth),
                "Q": float(child.Q),
                "visits": max(int(child.visits), 2),
                "expanded_recipe_ids": sorted(child.expanded_recipe_ids),
                "incoming_recipe_id": child.incoming_recipe_id,
                "token_usage": getattr(child, "token_usage", None),
                "stagnation_iteration": int(child.stagnation_iteration),
                "best_score": float(child.best_score),
                "score_mean": child.score_mean,
                "score_std": child.score_std,
            }
            # RecipeMCTS.backpropagate increments every ancestor once for
            # each newly created child. Apply the same update to disk
            # metadata without retaining those ancestors as Python objects.
            propagated_id = parent_id
            child_q = float(child.Q)
            while propagated_id is not None and propagated_id in records:
                propagated = records[propagated_id]
                propagated["Q"] = max(float(propagated.get("Q", child_q)),
                                       child_q)
                propagated["visits"] = int(propagated.get("visits", 1)) + 1
                propagated_id = propagated.get("parent_population_node_id")
        changed_ids = set()
        current_id = parent_id
        while current_id is not None and current_id in records:
            changed_ids.add(current_id)
            current_id = records[current_id].get(
                "parent_population_node_id")
        changed_ids.update(int(child.population_node_id) for child in children)
        self._set_pending_tree_state(root_id, records, changed_ids)

    def restore(self, checkpoint):
        """Restore tree metadata and RNG state for continued training."""
        from .resume import load_checkpoint, restore_tree
        state = load_checkpoint(checkpoint)
        restored_recipes = validate_recipes(state["recipes"])
        missing_expanders = set(restored_recipes).difference(
            self.expand_fn if isinstance(self.expand_fn, dict) else restored_recipes)
        if missing_expanders:
            raise ValueError(
                f"checkpoint recipes have no expander: {sorted(missing_expanders)}")
        self.recipes = {
            key: RecipeConfig(key, dict(values))
            for key, values in restored_recipes.items()
        }
        self.tree.recipes = dict(self.recipes)
        self.pop_size = int(state["pop_size"])
        self.selection_num = int(state["selection_num"])
        self.max_depth = int(state["max_depth"])
        self.tree.max_depth = self.max_depth
        if "max_sample_count" in state:
            saved_limit = state.get("max_sample_count")
            self.max_sample_count = (None if saved_limit is None
                                     else int(saved_limit))
        if "exploration_constant" in state:
            self.tree.exploration_constant = float(state["exploration_constant"])
        self.depth_balance_weight = float(state.get(
            "depth_balance_weight", state.get("depth_bias", self.depth_balance_weight)))
        self.tree.depth_balance_weight = self.depth_balance_weight
        if "elite_pool_size" in state:
            self.elite_pool_size = int(state.get("elite_pool_size") or 0)
            if self.elite_pool_size < 0:
                raise ValueError("checkpoint elite_pool_size must be >= 0")
            self.use_elite_pool = self.elite_pool_size > 0
            self._elite_pool = (self._load_elite_pool(state.get("elite_algorithm_ids"))
                            if self.use_elite_pool else [])
        adaptive_state = state.get("adaptive")
        if isinstance(adaptive_state, dict):
            self.adaptive_enabled = bool(
                adaptive_state.get("enabled", self.adaptive_enabled))
            self.adaptive_recipe_count = int(
                adaptive_state.get("recipes_per_expansion",
                                   self.adaptive_recipe_count))
            if self.adaptive_enabled and self.adaptive_router is None:
                self.adaptive_router = ContextualThompsonRouter(
                    self.recipes.keys(),
                    prior_precision=float(adaptive_state.get(
                        "prior_precision", 1.0)),
                    posterior_scale=float(adaptive_state.get(
                        "posterior_scale", 0.5)),
                )
            if self.adaptive_router is not None:
                self.adaptive_router.load_state_dict(
                    adaptive_state.get("router", {}))
        if self._disk_tree_mode:
            tree_state = self.store.load_tree_state()
            if tree_state.get("root_id") is None and state.get("nodes"):
                nodes = list(state.get("nodes", []))
                root_id = state.get("tree_root_id")
                if root_id is None:
                    root_id = nodes[0].get("population_node_id")
                self.store.write_tree_state(root_id, nodes)
            self.tree.root = None
            self._tree_root_id = self.store.load_tree_state().get("root_id")
            self._pending_tree_updates = {}
        else:
            root, _ = restore_tree(state)
            if root is None:
                raise ValueError("checkpoint does not contain a root node")
            self.tree.root = root
        self._next_population_node_id = int(
            state.get("next_population_node_id",
                      state.get("last_population_node_id", 0) + 1))
        self._next_algorithm_id = int(state["next_algorithm_id"])
        self._total_sample_count = int(state.get(
            "total_sample_count",
            max((item.get("sample_order", 0) for item in self._best_samples),
                default=0),
        ))
        self._best_score = float(state.get("best_score", self._best_score))
        saved_usage = state.get("cumulative_token_usage")
        if isinstance(saved_usage, dict):
            self._cumulative_token_usage = {
                "prompt_tokens": int(saved_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(saved_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(saved_usage.get("total_tokens", 0) or 0),
            }
        self.restore_random_state(state)

    def _write_checkpoint(self):
        path = os.path.join(self.store.directory,
                            f"checkpoint_{self._checkpoint_index:06d}.json")
        nodes = []
        if self._disk_tree_mode:
            # The historical tree index remains in tree_state.json. The
            # checkpoint stores only its location and root ID.
            tree_root_id = self._tree_root_id
            if tree_root_id is None:
                tree_root_id = self.store.load_tree_state().get("root_id")
        elif self.tree.root is not None:
            stack = [self.tree.root]
            while stack:
                node = stack.pop()
                nodes.append({
                    "population_node_id": node.population_node_id,
                    "parent_population_node_id": node.parent_population_node_id,
                    "depth": node.depth,
                    "Q": node.Q,
                    "visits": node.visits,
                    "children_population_node_ids": [x.population_node_id for x in node.children],
                    "expanded_recipe_ids": sorted(node.expanded_recipe_ids),
                    "incoming_recipe_id": node.incoming_recipe_id,
                    "token_usage": getattr(node, "token_usage", None),
                    "stagnation_iteration": getattr(
                        node, "stagnation_iteration", 0),
                    "best_score": getattr(node, "best_score", node.Q),
                    "score_mean": getattr(node, "score_mean", None),
                    "score_std": getattr(node, "score_std", None),
                })
                stack.extend(node.children)
            tree_root_id = self.tree.root.population_node_id
        else:
            tree_root_id = None
        state = {
            # The counter always points to the next unused ID.  Keep both
            # fields for compatibility with older checkpoints.
            "last_population_node_id": max(0, self._next_population_node_id - 1),
            "next_population_node_id": self._next_population_node_id,
            "next_algorithm_id": self._next_algorithm_id,
            "total_sample_count": self._total_sample_count,
            "max_sample_count": self.max_sample_count,
            "best_score": self._best_score,
            "max_depth": self.max_depth,
            "pop_size": self.pop_size,
            "selection_num": self.selection_num,
            "elite_pool_size": self.elite_pool_size if self.use_elite_pool else 0,
            "elite_pool_count": len(self._elite_pool),
            "elite_algorithm_ids": [
                getattr(item, "_recipe_algorithm_id", None)
                for item in self._elite_pool
            ],
            "exploration_constant": self.tree.exploration_constant,
            "depth_balance_weight": self.tree.depth_balance_weight,
            "nodes": nodes,
            "tree_root_id": tree_root_id,
            "recipes": {k: v.values for k, v in self.recipes.items()},
            "adaptive": {
                "enabled": self.adaptive_enabled,
                "recipes_per_expansion": self.adaptive_recipe_count,
                "prior_precision": (
                    self.adaptive_router.prior_precision
                    if self.adaptive_router is not None else None),
                "posterior_scale": (
                    self.adaptive_router.posterior_scale
                    if self.adaptive_router is not None else None),
                "router": (
                    self.adaptive_router.state_dict()
                    if self.adaptive_router is not None else None),
            },
            # Pickled/base64 states preserve exact Python and NumPy RNG state;
            # they are local checkpoint data, not executable input.
            "python_random_state": self._encode_state(random.getstate()),
            "numpy_random_state": (
                self._encode_state(np.random.get_state()) if np is not None else None
            ),
            # Cumulative LLM cost up to this checkpoint.  This is deliberately
            # the final field so it is easy to inspect in large checkpoint
            # files without opening node records.
            "cumulative_token_usage": dict(self._cumulative_token_usage),
        }
        self.store.write_checkpoint(path, state)

    @staticmethod
    def _encode_state(value):
        return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

    @staticmethod
    def _decode_state(value):
        return pickle.loads(base64.b64decode(value.encode("ascii")))

    def restore_random_state(self, checkpoint):
        """Restore RNG state from a checkpoint dictionary or file path."""
        if isinstance(checkpoint, (str, os.PathLike)):
            with open(checkpoint, encoding="utf-8") as f:
                checkpoint = json.load(f)
        encoded = checkpoint.get("python_random_state")
        if encoded:
            random.setstate(self._decode_state(encoded))
        encoded = checkpoint.get("numpy_random_state")
        if encoded and np is not None:
            np.random.set_state(self._decode_state(encoded))
