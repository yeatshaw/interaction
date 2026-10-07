from __future__ import annotations

import copy
import concurrent.futures
import itertools
import json
import math
import threading
from contextlib import nullcontext

from ...base import Function, SecureEvaluator, TextFunctionProgramConverter
from ..eoh.prompt import EoHPrompt
from ..partevo.base import Evoind
from ..partevo.clusterunit import ClusterUnit
from ..partevo.clustermanager import ClusterManager
from ..partevo.externalArchive import ExternalArchive
from ..partevo.prompt import PartEvoPrompt
from ..partevo.sampler import PartEvoSampler
from .mcts_recipe import MCTSRecipe
from .budget import SampleBudgetExhausted
from .population import RecipePopulation
from .refineevo_experience import RefineEvoExperienceManager
from .resume import function_from_record


class PartEvoBranch:
    """Serializable PartEvo state owned by one recipe-tree node."""

    def __init__(self, manager, experiences=None, algorithm_registry=None):
        self.manager = manager
        self.experiences = list(experiences or [])
        self._experience_lock = threading.RLock()
        self.algorithm_registry = dict(algorithm_registry or {
            int(getattr(item, "_recipe_algorithm_id")): item
            for item in self._collect_functions_from_manager(manager)
            if getattr(item, "_recipe_algorithm_id", None) is not None
        })
        self.last_token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

    @classmethod
    def create_root(cls, individuals, config):
        manager = ClusterManager(
            pop_size=config["pop_size"],
            n_clusters=config["partition_number"],
            intra_operators=tuple(config["operators"]),
            intra_operators_parent_num=config.get("operators_parent_num"),
            intra_operators_frequency=config.get("operators_frequency"),
            use_resource_tilt=config.get("use_resource_tilt", False),
            feature_type=tuple(config.get("feature_used", ("ast",))),
            bert_model_path=config.get("bert_model_path"),
            debug_flag=config.get("debug", False),
        )
        manager.population = list(individuals)
        manager.initial_population_clustering()
        for individual in manager.population:
            manager.external_archive.register(individual)
        return cls(manager, algorithm_registry={
            int(getattr(item, "_recipe_algorithm_id")): item
            for item in individuals
            if getattr(item, "_recipe_algorithm_id", None) is not None
        })

    @staticmethod
    def _collect_functions_from_manager(manager):
        result = []
        seen = set()
        stack = list(manager.population) + list(manager.next_pop)
        for unit in manager.cluster_units.values():
            stack.extend(item.function for item in unit.population)
        stack.extend(manager.external_archive.elites)
        stack.extend(manager.external_archive.hard_negatives)
        while stack:
            item = stack.pop()
            key = id(item)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def clone(self):
        return self.from_snapshot(self.snapshot(), self.manager_config())

    def manager_config(self):
        expanded = tuple(self.manager.cluster_units[0].intra_operators) \
            if self.manager.cluster_units else ()
        parent_num = dict(self.manager.intra_cluster_operators_parent_num)
        if "cn" in expanded and "cn" not in parent_num:
            parent_num["cn"] = parent_num.get("cc", 1)
        return {
            "pop_size": self.manager.pop_size,
            "partition_number": self.manager.n_clusters,
            "operators": expanded,
            "operators_parent_num": parent_num,
            "operators_frequency": None,
            "use_resource_tilt": self.manager.use_resource_tilt,
            "feature_used": tuple(self.manager.feature_type),
            "bert_model_path": self.manager.bert_model_path,
            "debug": self.manager.debug_flag,
        }

    @staticmethod
    def _collect_functions(branch):
        result = {}
        active = list(branch.manager.population) + list(branch.manager.next_pop)
        for unit in branch.manager.cluster_units.values():
            active.extend(item.function for item in unit.population)
        active.extend(branch.manager.external_archive.elites)
        active.extend(branch.manager.external_archive.hard_negatives)
        stack = [(item, 0) for item in active]
        while stack:
            item, relation_depth = stack.pop()
            algorithm_id = getattr(item, "_recipe_algorithm_id", None)
            key = ("id", int(algorithm_id)) if algorithm_id is not None \
                else ("code", str(item))
            if key in result:
                continue
            result[key] = item
            # Keep direct prompt parents only. Older ancestors are resolved
            # from persisted node records when a node is loaded.
            if relation_depth == 0:
                stack.extend((parent, 1) for parent in
                             getattr(item, "_recipe_parent_functions", ()) or ())
        return list(result.values())

    @staticmethod
    def _record(item):
        from .persistence import RecipeStore
        return RecipeStore._algorithm_record(item)

    def snapshot(self):
        functions = self._collect_functions(self)
        records = {str(getattr(item, "_recipe_algorithm_id", id(item))):
                   self._record(item) for item in functions}
        by_id = {
            str(getattr(item, "_recipe_algorithm_id", id(item))): item
            for item in functions
        }

        def ids(items):
            return [str(getattr(item, "_recipe_algorithm_id", id(item)))
                    for item in items]

        archive = self.manager.external_archive
        return {
            "population": ids(self.manager.population),
            "next_pop": ids(self.manager.next_pop),
            "generation": int(self.manager.generation),
            "global_best_id": (str(getattr(self.manager.global_best_ind,
                                            "_recipe_algorithm_id", ""))
                               if self.manager.global_best_ind else None),
            "algorithms": list(records.values()),
            "clusters": [
                {
                    "cluster_id": int(cluster_id),
                    "algorithm_ids": ids([item.function for item in unit.population]),
                    "operator_position": int(getattr(unit, "_operator_position", 0)),
                    "non_improvement": int(unit.cumulative_Non_improvement_count),
                    "history_best_score": float(unit._history_best_score),
                }
                for cluster_id, unit in self.manager.cluster_units.items()
            ],
            "archive": {
                "elites": ids(archive.elites),
                "hard_negatives": ids(archive.hard_negatives),
                "cached_summary": archive.global_summary,
                "request_counter": int(archive._request_counter),
                "has_unsummarized_updates": bool(archive._has_unsummarized_updates),
            },
            "experiences": copy.deepcopy(self.experiences),
        }

    @classmethod
    def from_snapshot(cls, snapshot, config, store=None):
        records = {int(item["algorithm_id"]): item
                   for item in snapshot.get("algorithms", [])
                   if item.get("algorithm_id") is not None}
        parent_ids = {
            int(parent_id)
            for item in records.values()
            for parent_id in item.get("parent_algorithm_ids", [])
            if parent_id is not None
        }
        if store is not None:
            missing = store.load_algorithm_records(parent_ids - set(records))
            records.update({int(key): value for key, value in missing.items()})
        functions = {algorithm_id: function_from_record(record)
                     for algorithm_id, record in records.items()}
        for function in functions.values():
            function._recipe_parent_functions = tuple(
                functions[parent_id]
                for parent_id in getattr(function, "_eoh_parent_ids", ())
                if parent_id in functions
            )

        def resolve(values):
            return [functions[int(value)] for value in values
                    if int(value) in functions]

        manager = ClusterManager(
            pop_size=config["pop_size"],
            n_clusters=config["partition_number"],
            intra_operators=tuple(config["operators"]),
            intra_operators_parent_num=config.get("operators_parent_num"),
            intra_operators_frequency=config.get("operators_frequency"),
            use_resource_tilt=config.get("use_resource_tilt", False),
            feature_type=tuple(config.get("feature_used", ("ast",))),
            bert_model_path=config.get("bert_model_path"),
            debug_flag=config.get("debug", False),
        )
        manager.population = resolve(snapshot.get("population", []))
        manager.next_pop = resolve(snapshot.get("next_pop", []))
        manager.generation = int(snapshot.get("generation", 0))
        manager.cluster_units = {}
        for cluster in snapshot.get("clusters", []):
            cluster_id = int(cluster["cluster_id"])
            unit = ClusterUnit(
                cluster_id=cluster_id,
                max_pop_size=manager.pop_size,
                intra_operators=manager.intra_cluster_operators,
                intra_operators_parent_num=manager.intra_cluster_operators_parent_num,
                pop=[Evoind(function=item, cluster_id=cluster_id)
                     for item in resolve(cluster.get("algorithm_ids", []))],
            )
            unit._operator_position = int(cluster.get("operator_position", 0))
            unit.cumulative_Non_improvement_count = int(
                cluster.get("non_improvement", 0))
            unit._history_best_score = float(
                cluster.get("history_best_score", float("-inf")))
            manager.cluster_units[cluster_id] = unit
        manager.is_initialized = bool(manager.cluster_units)
        manager.global_best_ind = next(
            (functions[int(snapshot["global_best_id"])]
             for _ in [0]
             if snapshot.get("global_best_id") is not None
             and int(snapshot["global_best_id"]) in functions),
            None,
        )
        archive = manager.external_archive
        archive.elites = resolve(snapshot.get("archive", {}).get("elites", []))
        archive.hard_negatives = resolve(
            snapshot.get("archive", {}).get("hard_negatives", []))
        archive._cached_summary = snapshot.get("archive", {}).get(
            "cached_summary", "")
        archive._request_counter = int(snapshot.get("archive", {}).get(
            "request_counter", 0))
        archive._has_unsummarized_updates = bool(
            snapshot.get("archive", {}).get("has_unsummarized_updates", False))
        return cls(manager, snapshot.get("experiences", []),
                   algorithm_registry=functions)


class MCTSPartEvoRecipe(MCTSRecipe):
    """Recipe-MCTS whose state at every node is an independent PartEvo run."""

    def __init__(self, llm, evaluation, info, recipes, *, pop_size=16,
                 partition_number=4, operators=("re", "se", "cn", "lge"),
                 operators_parent_num=None, operators_frequency=None,
                 num_samplers=16, num_evaluators=16, max_round_attempts=3,
                 feature_used=("ast",), use_resource_tilt=False,
                 bert_model_path=None, debug=False, **kwargs):
        self._disk_tree_mode = True
        self._info = dict(info)
        self._evaluation = evaluation
        self._llm = llm
        self._operators = tuple(operators)
        self._branch_config = {
            "pop_size": int(pop_size),
            "partition_number": int(partition_number),
            "operators": self._operators,
            "operators_parent_num": operators_parent_num,
            "operators_frequency": operators_frequency,
            "feature_used": tuple(feature_used),
            "use_resource_tilt": bool(use_resource_tilt),
            "bert_model_path": bert_model_path,
            "debug": bool(debug),
        }
        self._num_samplers = max(1, int(num_samplers))
        self._num_evaluators = max(1, int(num_evaluators))
        self._max_round_attempts = max(1, int(max_round_attempts))
        self._template_program = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program)
        self._function_to_evolve = TextFunctionProgramConverter.text_to_function(
            evaluation.template_program)
        self._sampler = PartEvoSampler(llm, evaluation.template_program)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug)
        self._reflection_lock = threading.RLock()
        self._archive_lock = threading.RLock()
        self._experience_embedding_model = None
        super().__init__(
            recipes=recipes,
            pop_size=pop_size,
            selection_num=1,
            expand_fn={recipe_id: True for recipe_id in recipes},
            **kwargs,
        )
        self._token_usage_lock = threading.RLock()
        self._diagnostic_lock = threading.RLock()
        self._diagnostic_path = (
            self.store.directory + "/partevo_diagnostics.json")
        try:
            with open(self._diagnostic_path, encoding="utf-8") as stream:
                self._diagnostics = json.load(stream)
        except (OSError, json.JSONDecodeError):
            self._diagnostics = {
                "counts": {}, "recipes": {}, "rounds": [],
                "recent_failures": [],
            }
        self._diagnostics.setdefault("counts", {})
        self._diagnostics.setdefault("recipes", {})
        self._diagnostics.setdefault("rounds", [])
        self._diagnostics.setdefault("recent_failures", [])

    def _diagnose(self, category, recipe_id=None, *, detail=None,
                  data=None, persist=False):
        with self._diagnostic_lock:
            counts = self._diagnostics["counts"]
            counts[category] = int(counts.get(category, 0)) + 1
            if recipe_id is not None:
                recipe = self._diagnostics["recipes"].setdefault(
                    str(recipe_id), {"counts": {}, "rounds": []})
                recipe_counts = recipe.setdefault("counts", {})
                recipe_counts[category] = int(
                    recipe_counts.get(category, 0)) + 1
            if detail:
                failure = {
                    "category": category,
                    "recipe_id": recipe_id,
                    "detail": str(detail)[:500],
                }
                failures = self._diagnostics["recent_failures"]
                failures.append(failure)
                del failures[:-50]
            if data is not None:
                record = dict(data)
                self._diagnostics["rounds"].append(record)
                del self._diagnostics["rounds"][:-200]
                if recipe_id is not None:
                    recipe = self._diagnostics["recipes"].setdefault(
                        str(recipe_id), {"counts": {}, "rounds": []})
                    recipe_rounds = recipe.setdefault("rounds", [])
                    recipe_rounds.append(record)
                    del recipe_rounds[:-100]
            if persist:
                self._diagnostics["total_sample_count"] = (
                    self._total_sample_count)
                self._diagnostics["max_sample_count"] = self.max_sample_count
                self.store.write_checkpoint(
                    self._diagnostic_path, self._diagnostics)

    def _set_stop_diagnostic(self, reason):
        with self._diagnostic_lock:
            self._diagnostics["stop_reason"] = reason
            self._diagnostics["total_sample_count"] = self._total_sample_count
            self._diagnostics["max_sample_count"] = self.max_sample_count
            self._diagnostics["next_algorithm_id"] = self._next_algorithm_id
            self._diagnostics["next_population_node_id"] = (
                self._next_population_node_id)
            self.store.write_checkpoint(
                self._diagnostic_path, self._diagnostics)

    def _record_token_usage(self, usage):
        if not isinstance(usage, dict):
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion)
                    or prompt + completion)
        with self._token_usage_lock:
            self._cumulative_token_usage["prompt_tokens"] += prompt
            self._cumulative_token_usage["completion_tokens"] += completion
            self._cumulative_token_usage["total_tokens"] += total
            self.store.write_token_usage(self._cumulative_token_usage)

    def _append_convergence_record(self, individual, population_node):
        added = super()._append_convergence_record(individual, population_node)
        if added:
            running_best = float("-inf")
            for record in sorted(
                    self._convergence_records,
                    key=lambda item: int(item.get("sample_order") or 0)):
                score = record.get("score")
                if score is not None:
                    running_best = max(running_best, float(score))
                record["best_so_far"] = (
                    running_best if running_best != float("-inf") else None)
        return added

    def _write_convergence_outputs(self, plot=False):
        if not self._convergence_records:
            return
        if self._convergence_dirty:
            self.store.write_convergence_records(
                self._convergence_records,
                x_axis="sample_order",
                y_axis="best_so_far",
            )
            self._last_convergence_write_count = len(self._convergence_records)
            self._convergence_dirty = False
        if plot:
            try:
                self.store.write_convergence_plot(
                    self._convergence_records,
                    x_axis="sample_order",
                    y_axis="best_so_far",
                )
            except Exception as exc:
                print(f"WARNING: failed to write convergence plot: {exc}",
                      flush=True)

    def _write_checkpoint(self):
        with self._token_usage_lock:
            super()._write_checkpoint()
            self.store.write_token_usage(self._cumulative_token_usage)

    def set_experience_embedding_model(self, embedding_model):
        self._experience_embedding_model = embedding_model

    def close(self):
        self._sampler.llm = self._llm

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
                    "use a new log directory or checkpoint")
            self.initialize(initial_individuals, initial_samples_recorded,
                            initial_token_usage)

        stop_reason = "tree_exhausted"
        while True:
            if self.sample_budget_exhausted():
                with self._tree_lock:
                    self._log_sample_limit_once_locked()
                stop_reason = "max_sample_count_reached"
                break
            node = self._select_disk_tree_node()
            if node is None:
                stop_reason = "no_selectable_node"
                break
            if node.depth >= self.max_depth:
                stop_reason = "max_depth_reached"
                break
            branch, experiences = self._load_node_state(node.population_node_id)
            samples_before_expansion = self._total_sample_count
            children = self.expand_node(node, branch, experiences)
            expansion_samples = self._total_sample_count - samples_before_expansion
            self._diagnose(
                "node_expansion", data={
                    "node_id": node.population_node_id,
                    "depth": node.depth,
                    "samples_admitted": expansion_samples,
                    "children_created": len(children),
                }, persist=True)
            self._sync_disk_tree(node, children)
            if self._tree_checkpoint_pending:
                # A parallel batch may cross the 100-node boundary while
                # other workers are still appending records. Flush the
                # remainder before writing a checkpoint so its index never
                # points to an in-memory-only node.
                self.store.flush()
                self._commit_pending_tree_state()
                self._checkpoint_index += 1
                self._write_checkpoint()
                self._tree_checkpoint_pending = False
            has_children = bool(children)
            del children, branch, node
            if self.sample_budget_exhausted():
                with self._tree_lock:
                    self._log_sample_limit_once_locked()
                stop_reason = "max_sample_count_reached"
                break
            if not has_children:
                # A PartEvo round can fail to complete a generation when the
                # LLM returns too few valid candidates.  If evaluation budget
                # was consumed, keep the node selectable and retry the pending
                # recipes instead of ending the whole experiment early.  When
                # no candidate entered evaluation, stop to avoid an empty-loop
                # caused by a persistent parsing/configuration error.
                if self._total_sample_count > samples_before_expansion:
                    continue
                stop_reason = "no_valid_candidate_in_expansion"
                break

        self._set_stop_diagnostic(stop_reason)
        self.store.flush()
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self._write_convergence_outputs(plot=True)
        usage = dict(self._cumulative_token_usage)
        self.store.write_token_usage(usage)
        print(
            "Total LLM tokens: "
            f"input={usage['prompt_tokens']}, "
            f"output={usage['completion_tokens']}, "
            f"total={usage['total_tokens']}",
            flush=True,
        )
        return None

    @staticmethod
    def _capture(llm):
        capture = getattr(llm, "capture_token_usage", None)
        return capture() if callable(capture) else nullcontext({})

    @staticmethod
    def _reference_copy(function):
        shell = copy.copy(function)
        parents = []
        for parent in getattr(function, "_recipe_parent_functions", ()):
            parent_shell = copy.copy(parent)
            parent_shell._recipe_parent_functions = ()
            parents.append(parent_shell)
        shell._recipe_parent_functions = tuple(parents)
        return shell

    def _merge_usage(self, usages):
        values = [item for item in usages if isinstance(item, dict)]
        merge = getattr(self._llm, "merge_token_usage", None)
        if callable(merge):
            result = merge(*values) if values else {}
        else:
            result = {
                "prompt_tokens": sum(int(x.get("prompt_tokens", 0)) for x in values),
                "completion_tokens": sum(int(x.get("completion_tokens", 0)) for x in values),
            }
        result = dict(result or {})
        result["total_tokens"] = int(result.get("prompt_tokens", 0)) + int(
            result.get("completion_tokens", 0))
        return result

    def initialize(self, individuals, initial_samples_recorded=False,
                   initial_token_usage=None):
        population = RecipePopulation(copy.deepcopy(individuals), self.pop_size)
        self._assign_algorithm_ids(population.individuals)
        branch = PartEvoBranch.create_root(population.individuals,
                                           self._branch_config)
        best, mean, std = self._population_stats(population)
        root_id = self._next_population_node_id
        self._next_population_node_id += 1
        root = self.tree.root = self._new_node(
            root_id, best, depth=0, best_score=best,
            score_mean=mean, score_std=std, token_usage=initial_token_usage)
        self._diagnose(
            "initialization_persisted",
            data={
                "event": "initialization",
                "population_node_id": int(root_id),
                "input_algorithms": len(individuals),
                "retained_population": len(population.individuals),
                "total_sample_count": self._total_sample_count,
            },
            persist=True,
        )
        if not initial_samples_recorded:
            self._record_evaluated_candidates(population.individuals, root)
        self._record_token_usage(initial_token_usage)
        self._write_branch_node(root, branch, None, initial_token_usage)
        self.store.flush()
        self._set_pending_tree_state(root_id, [self._tree_record(root)])
        self.store.flush()
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self._tree_checkpoint_pending = False
        self.tree.root = None
        return root

    def _new_node(self, node_id, best, **kwargs):
        from .mcts import RecipeNode
        return RecipeNode(node_id, best, **kwargs)

    def _write_branch_node(self, node, branch, recipe_id, token_usage=None):
        population = RecipePopulation(
            branch.manager.population, self.pop_size, branch.manager.generation)
        flushed = self.store.add(
            node.population_node_id,
            node.parent_population_node_id,
            node.depth,
            recipe_id,
            population.generation,
            population,
            branch.experiences,
            token_usage=token_usage,
            extra={"partevo_state": branch.snapshot()},
        )
        if flushed is not None:
            # The parent/child index is committed after all parallel recipe
            # branches have completed, so checkpointing is deferred.
            self._tree_checkpoint_pending = True

    @staticmethod
    def _tree_record(node):
        return {
            "population_node_id": int(node.population_node_id),
            "parent_population_node_id": (
                int(node.parent_population_node_id)
                if node.parent_population_node_id is not None else None
            ),
            "depth": int(node.depth),
            "Q": float(node.Q),
            "visits": int(node.visits),
            "children_population_node_ids": [],
            "expanded_recipe_ids": sorted(node.expanded_recipe_ids),
            "incoming_recipe_id": node.incoming_recipe_id,
            "token_usage": getattr(node, "token_usage", None),
            "stagnation_iteration": int(node.stagnation_iteration),
            "best_score": float(node.best_score),
            "score_mean": node.score_mean,
            "score_std": node.score_std,
        }

    def _sync_disk_tree(self, parent, children):
        payload = self._disk_tree_payload()
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["population_node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        root_id = payload.get("root_id")
        if root_id is None:
            root_id = int(parent.population_node_id)
        parent_id = int(parent.population_node_id)
        parent_record = by_id.setdefault(parent_id, self._tree_record(parent))
        parent_record["expanded_recipe_ids"] = sorted(
            set(parent_record.get("expanded_recipe_ids", [])) |
            {child.incoming_recipe_id for child in children
             if child.incoming_recipe_id is not None}
        )
        parent_record["children_population_node_ids"] = sorted(
            {int(item["population_node_id"]) for item in by_id.values()
             if item.get("parent_population_node_id") == parent_id}
            | {int(child.population_node_id) for child in children}
        )
        changed_ids = {parent_id}
        for child in children:
            child_record = self._tree_record(child)
            child_record["visits"] = max(int(child_record["visits"]), 2)
            by_id[int(child.population_node_id)] = child_record
            changed_ids.add(int(child.population_node_id))
        for child in children:
            child_q = float(child.Q)
            current_id = parent_id
            while current_id in by_id:
                current = by_id[current_id]
                current["Q"] = max(float(current.get("Q", child_q)), child_q)
                current["visits"] = int(current.get("visits", 1)) + 1
                changed_ids.add(int(current_id))
                current_id = current.get("parent_population_node_id")
        self._set_pending_tree_state(root_id, by_id, changed_ids)

    def _load_node_state(self, node_id):
        record = self.store.load_node(node_id)
        branch_state = record.get("partevo_state")
        if not branch_state:
            raise KeyError(f"PartEvo state for population node {node_id} is not persisted")
        branch = PartEvoBranch.from_snapshot(
            branch_state, self._branch_config, store=self.store)
        return branch, list(branch.experiences)

    @staticmethod
    def _format_experiences(experiences):
        if not experiences:
            return ""
        success, failure, reference = [], [], []
        for item in experiences:
            text = item if isinstance(item, str) else str(item)
            status = item.get("is_success") if isinstance(item, dict) else None
            target = success if status is True else failure if status is False else reference
            target.append(text)
        return (
            "### DESIGN EXPERIENCES FROM PREVIOUS ATTEMPTS\n"
            "#### SUCCESSFUL EXPERIENCES\n" + "\n".join(success or ["None"]) + "\n"
            "#### FAILED EXPERIENCES\n" + "\n".join(failure or ["None"]) + "\n"
            "#### REFERENCE-ONLY EXPERIENCES\n" + "\n".join(reference or ["None"])
        )

    @staticmethod
    def _objective_improved(child, parents):
        parent_scores = [float(item.score) for item in parents
                         if getattr(item, "score", None) is not None]
        child_score = getattr(child, "score", None)
        return (child_score is not None
                and (not parent_scores
                     or float(child_score) > max(parent_scores)))

    def _reflection(self, refs, recipe, branch, operator,
                    experience_manager=None, experience_snapshot=None):
        values = recipe.values
        if values.get("refineevo_experience", False):
            manager = experience_manager
            if manager is None:
                manager = RefineEvoExperienceManager(
                    reflector_llm=self._llm,
                    embedding_model=self._experience_embedding_model,
                    top_k=int(values.get("experience_top_k", 3)),
                )
            experiences = list(experience_snapshot or [])
            retrieved = manager.retrieve(refs, operator, experiences)
            fresh = manager.distill(refs, operator)
            retrieved = retrieved if isinstance(retrieved, list) else []
            fresh = fresh if isinstance(fresh, list) else ([fresh] if fresh else [])
            return self._format_experiences(retrieved + fresh) or None, retrieved, fresh

        parent_groups = [list(getattr(ref, "_recipe_parent_functions", ()))
                         for ref in refs]
        prompt = EoHPrompt.get_prompt_reflection(
            refs,
            parents=parent_groups,
            info=self._info,
            parent_info_flag=values.get("parent_info", True),
            best_worst_flag=values.get("best_worst", False),
            fitness_flag=values.get("fitness", 2),
            avg_fitness_flag=values.get("avg_fitness", False),
            check_reflection_flag=values.get("check_guidance", True),
            population=branch.manager.population,
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
        response = self._llm.draw_sample(prompt)
        suggestion = response.strip() if isinstance(response, str) else None
        return suggestion, [], []

    @staticmethod
    def _append_guidance(messages, suggestion):
        if not suggestion:
            return messages
        messages = copy.deepcopy(messages)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text":
                          "### Recipe Reflection\n" + str(suggestion)}],
        })
        return messages

    def _operator_prompt(self, refs, operator, suggestion, branch):
        if operator == "re":
            return PartEvoPrompt.get_prompt_re(
                self._info["task_description"], refs[0],
                self._function_to_evolve, suggestion or "")
        if operator == "se":
            with self._archive_lock:
                summary, requires_update, samples = (
                    branch.manager.external_archive.fetch_summary_context())
            if requires_update and samples:
                summary_prompt = PartEvoPrompt.get_prompt_summary(
                    self._info["task_description"], self._function_to_evolve,
                    samples)
                generated = self._llm.draw_sample(summary_prompt)
                branch.manager.external_archive.update_global_summary(generated)
                if generated and generated.strip():
                    summary = generated.strip()
            messages = PartEvoPrompt.get_prompt_se(
                self._info["task_description"], refs[0],
                self._function_to_evolve, summary or "")
            return self._append_guidance(messages, suggestion)
        if operator in ("cn", "cc"):
            messages = PartEvoPrompt.get_prompt_cn(
                self._info["task_description"], refs,
                self._function_to_evolve)
            return self._append_guidance(messages, suggestion)
        if operator == "lge":
            messages = PartEvoPrompt.get_prompt_lge(
                self._info["task_description"], refs,
                self._function_to_evolve)
            return self._append_guidance(messages, suggestion)
        raise ValueError(f"Unsupported PartEvo operator: {operator}")

    def _prepare_candidate(self, task, branch, recipe):
        refs = task["refs"]
        operator = task["operator"]
        recipe_id = task.get("recipe_id")
        self._diagnose("candidate_started", recipe_id)
        reflection_usage = {}
        evolution_usage = {}
        suggestion = None
        retrieved = []
        fresh = []
        function = None
        program = None
        thought = None
        response = None
        prompt = None
        generation_exception = False
        preparation_stage = "reflection"
        try:
            self._diagnose("reflection_attempt", recipe_id)
            with self._capture(self._llm) as reflection_usage:
                suggestion, retrieved, fresh = self._reflection(
                    refs, recipe, branch, operator,
                    experience_manager=task.get("experience_manager"),
                    experience_snapshot=task.get("experience_snapshot"),
                )
                if suggestion is None:
                    self._diagnose("reflection_empty", task.get("recipe_id"))
                preparation_stage = "prompt_build"
                prompt = self._operator_prompt(
                    refs, operator, suggestion, branch)
        except Exception as exc:
            self._diagnose(
                f"{preparation_stage}_error", task.get("recipe_id"),
                detail=f"{type(exc).__name__}: {exc}")
        if prompt is not None:
            try:
                self._diagnose("algorithm_generation_attempt", recipe_id)
                with self._capture(self._llm) as evolution_usage:
                    thought, function, response = self._sampler.get_thought_and_function(
                        prompt="", messages=prompt)
            except Exception as exc:
                generation_exception = True
                self._diagnose(
                    "llm_generation_error", task.get("recipe_id"),
                    detail=f"{type(exc).__name__}: {exc}")
            if thought is None and not generation_exception:
                self._diagnose("thought_parse_failure", recipe_id)
            if function is None and not generation_exception:
                self._diagnose(
                    "function_parse_failure", recipe_id,
                    detail=(f"response_length={len(response)}"
                            if isinstance(response, str) else "response is not text"))
            elif thought is not None and function is not None:
                try:
                    program = TextFunctionProgramConverter.function_to_program(
                        function, self._template_program)
                except Exception as exc:
                    self._diagnose(
                        "program_conversion_error", recipe_id,
                        detail=f"{type(exc).__name__}: {exc}")
                else:
                    if program is None:
                        self._diagnose(
                            "program_conversion_error", recipe_id,
                            detail="function_to_program returned None")
                    else:
                        self._diagnose("program_prepared", recipe_id)
        token_usage = self._merge_usage(
            [reflection_usage, evolution_usage])
        self._record_token_usage(token_usage)
        recorder = task.get("record_token_usage")
        if recorder is not None:
            recorder(token_usage)
        if thought is None or function is None or program is None:
            return None
        function.algorithm = thought
        function.operator = "cn" if operator == "cc" else operator
        function.parents = [
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs]
        function._eoh_parent_ids = tuple(function.parents)
        function._recipe_id = recipe.recipe_id
        function._recipe_sample_order = task.get("sample_order")
        function._eoh_generation_suggestion = suggestion
        function._eoh_experience = (
            suggestion if recipe.values.get("refineevo_experience", False)
            else None)
        function.response = response
        function.prompt = prompt
        function._recipe_token_usage = token_usage
        function._recipe_parent_functions = tuple(
            self._reference_copy(ref) for ref in refs)
        return {"function": function, "program": program,
                "cluster_id": task["cluster_id"],
                "recipe_id": task.get("recipe_id"),
                "refs": refs,
                "retrieved_experiences": retrieved,
                "fresh_experiences": fresh,
                "token_usage": function._recipe_token_usage}

    def admit_evaluation_sample(self):
        """Count one candidate immediately before it enters evaluation.

        PartEvo generation can produce invalid programs, so its global budget
        is admitted after parsing rather than reserved before the LLM call.
        The lock makes this safe for concurrent initialization and branches.
        """
        with self._tree_lock:
            if self.sample_budget_exhausted():
                self._log_sample_limit_once_locked()
                raise SampleBudgetExhausted(
                    f"sample limit reached: {self._total_sample_count}/"
                    f"{self.max_sample_count}")
            self._total_sample_count += 1
            return self._total_sample_count

    def _admit_candidates(self, candidates):
        admitted = []
        for item in candidates:
            try:
                sample_order = self.admit_evaluation_sample()
            except SampleBudgetExhausted:
                break
            item["sample_order"] = sample_order
            item["function"]._recipe_sample_order = sample_order
            admitted.append(item)
        return admitted

    def _evaluate_candidates(self, candidates):
        if not candidates:
            return []
        evaluated = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_evaluators) as executor:
            futures = {
                executor.submit(
                    self._evaluator.evaluate_program_record_time,
                    item["program"]): item
                for item in candidates
            }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, tuple):
                        value, eval_time = result
                    else:
                        value, eval_time = result, None
                    if isinstance(value, dict):
                        item["function"].all_ins_performance = value.get(
                            "all_ins_performance")
                        item["function"].list_performance = value.get(
                            "list_performance")
                        value = value.get("score")
                    if value is None or not math.isfinite(float(value)):
                        self._diagnose(
                            "evaluation_invalid_score",
                            item.get("recipe_id"),
                            detail=f"score={value!r}")
                        continue
                    item["function"].score = float(value)
                    item["function"].evaluate_time = eval_time
                    evaluated.append(item)
                    self._diagnose("evaluation_success", item.get("recipe_id"))
                except Exception as exc:
                    self._diagnose(
                        "evaluation_error", item.get("recipe_id"),
                        detail=f"{type(exc).__name__}: {exc}")
        return evaluated

    def _evolve_round(self, branch, recipe, child_node_id, depth):
        target = self.pop_size
        target_generation = branch.manager.generation + 1
        success = []
        attempts = 0
        prepared_count = 0
        admitted_count = 0
        evaluated_count = 0
        rejected_count = 0
        usages = []
        usage_lock = threading.RLock()

        def record_round_usage(usage):
            with usage_lock:
                usages.append(usage)
        experience_manager = None
        experience_snapshot = None
        if recipe.values.get("refineevo_experience", False):
            experience_manager = RefineEvoExperienceManager(
                reflector_llm=self._llm,
                embedding_model=self._experience_embedding_model,
                top_k=int(recipe.values.get("experience_top_k", 3)),
            )
            with branch._experience_lock:
                experience_snapshot = copy.deepcopy(branch.experiences)
            experience_manager.begin_node(experience_snapshot)
        try:
            while (branch.manager.generation < target_generation
                   and len(success) < target * self._max_round_attempts
                   and attempts < target * self._max_round_attempts):
                if self.sample_budget_exhausted():
                    break
                task_batch = []
                batch_target = target
                selections_without_parent = 0
                for _ in range(batch_target):
                    refs, operator, cluster_id = branch.manager.select_parent()
                    if not refs or not operator or cluster_id < 0:
                        selections_without_parent += 1
                        continue
                    task_batch.append({
                        "refs": [self._reference_copy(ref) for ref in refs],
                        "operator": operator,
                        "cluster_id": cluster_id,
                        "recipe_id": recipe.recipe_id,
                        "experience_manager": experience_manager,
                        "experience_snapshot": experience_snapshot,
                        "record_token_usage": record_round_usage,
                    })
                    attempts += 1
                if not task_batch:
                    break
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self._num_samplers, len(task_batch))) as executor:
                    futures = [executor.submit(
                        self._prepare_candidate, task, branch, recipe)
                        for task in task_batch]
                    prepared = [future.result() for future in futures]
                prepared = [item for item in prepared if item is not None]
                prepared_count += len(prepared)
                admitted = self._admit_candidates(prepared)
                admitted_count += len(admitted)
                evaluated = self._evaluate_candidates(admitted)
                evaluated_count += len(evaluated)
                accepted_in_batch = 0
                for item in evaluated:
                    function = item["function"]
                    if recipe.values.get("refineevo_experience", False):
                        improved = self._objective_improved(
                            function, item["refs"])
                        retrieved_ids = {
                            value.get("experience_id")
                            for value in item["retrieved_experiences"]
                            if isinstance(value, dict)
                            and value.get("experience_id")
                        }
                        with branch._experience_lock:
                            retained = []
                            for value in branch.experiences:
                                if not isinstance(value, dict):
                                    retained.append(value)
                                    continue
                                if value.get("experience_id") in retrieved_ids:
                                    value["score"] = int(value.get("score", 0)) + (
                                        1 if improved else -1)
                                if int(value.get("score", 0)) >= 0:
                                    retained.append(value)
                            retained.extend(copy.deepcopy(
                                item["fresh_experiences"]))
                            branch.experiences[:] = retained
                    self.record_evaluated_sample(function, child_node_id, depth)
                    accepted = branch.manager.register_function(
                        function, from_which_cluster=item["cluster_id"])
                    if accepted:
                        success.append(function)
                        accepted_in_batch += 1
                        self._diagnose("population_accepted", recipe.recipe_id)
                    else:
                        rejected_count += 1
                        self._diagnose("population_rejected", recipe.recipe_id)
                self._diagnose(
                    "candidate_batch",
                    recipe.recipe_id,
                    data={
                        "event": "candidate_batch",
                        "recipe_id": recipe.recipe_id,
                        "population_node_id": int(child_node_id),
                        "tasks_submitted": len(task_batch),
                        "selections_without_parent": selections_without_parent,
                        "programs_prepared": len(prepared),
                        "admitted_to_evaluation": len(admitted),
                        "finite_evaluations": len(evaluated),
                        "population_accepted": accepted_in_batch,
                        "population_rejected": len(evaluated) - accepted_in_batch,
                        "sample_count": self._total_sample_count,
                        "sample_budget_exhausted": self.sample_budget_exhausted(),
                    },
                    persist=True,
                )
                if self.sample_budget_exhausted():
                    break
        finally:
            if experience_manager is not None:
                experience_manager.end_node()

        # PartEvo's normal trigger is next_pop >= pop_size.  The batch above
        # deliberately registers only after all generation/evaluation work is
        # complete, so the same global survival trigger remains intact.
        branch.last_token_usage = self._merge_usage(usages)
        completed = branch.manager.generation >= target_generation
        round_record = {
            "event": "recipe_round",
            "recipe_id": recipe.recipe_id,
            "population_node_id": int(child_node_id),
            "target_generation": int(target_generation),
            "completed": completed,
            "generation_after": int(branch.manager.generation),
            "attempts": int(attempts),
            "prepared": int(prepared_count),
            "admitted_to_evaluation": int(admitted_count),
            "evaluated_with_finite_score": int(evaluated_count),
            "population_accepted": int(len(success)),
            "population_rejected": int(rejected_count),
            "next_pop_size": len(branch.manager.next_pop),
            "population_size": len(branch.manager.population),
            "sample_count": self._total_sample_count,
            "sample_budget_exhausted": self.sample_budget_exhausted(),
        }
        self._diagnose(
            "generation_completed" if completed else "generation_incomplete",
            recipe.recipe_id, data=round_record, persist=True)
        print(
            f"PartEvo recipe round {recipe.recipe_id}: "
            f"completed={completed}, attempts={attempts}, "
            f"prepared={prepared_count}, evaluated={evaluated_count}, "
            f"accepted={len(success)}, next_pop="
            f"{len(branch.manager.next_pop)}/{target}",
            flush=True,
        )
        return completed

    def expand_node(self, node, branch, experiences=None):
        pending = [(recipe_id, recipe) for recipe_id, recipe in self.recipes.items()
                   if recipe_id not in node.expanded_recipe_ids]
        if not pending:
            return []
        adaptive_state = None
        if self.adaptive_enabled:
            parent_population = RecipePopulation(
                branch.manager.population, self.pop_size,
                branch.manager.generation)
            adaptive_state = self._adaptive_state(node, parent_population)
            selected_ids, _ = self.adaptive_router.select_top_k(
                adaptive_state,
                [recipe_id for recipe_id, _ in pending],
                k=min(self.adaptive_recipe_count, len(pending)),
            )
            selected = set(selected_ids)
            pending = [(recipe_id, recipe) for recipe_id, recipe in pending
                       if recipe_id in selected]
        with self._tree_lock:
            child_ids = {
                recipe_id: self._next_population_node_id + index
                for index, (recipe_id, _) in enumerate(pending)
            }
            self._next_population_node_id += len(pending)

        children = []

        def expand_one(recipe_id, recipe):
            branch_copy = branch.clone()
            completed = self._evolve_round(
                branch_copy, recipe, child_ids[recipe_id], node.depth + 1)
            if not completed:
                print(
                    f"Skipped incomplete PartEvo node recipe={recipe_id}: "
                    f"generation={branch_copy.manager.generation}, "
                    f"next_pop={len(branch_copy.manager.next_pop)}/{self.pop_size}",
                    flush=True,
                )
                return
            population = RecipePopulation(
                branch_copy.manager.population, self.pop_size,
                branch_copy.manager.generation)
            best, mean, std = self._population_stats(population)
            with self._tree_lock:
                child = self._new_node(
                    child_ids[recipe_id], best, depth=node.depth + 1,
                    parent=node, incoming_recipe_id=recipe_id,
                    token_usage=branch_copy.last_token_usage,
                    stagnation_iteration=(
                        0 if best > node.best_score
                        else node.stagnation_iteration + 1),
                    best_score=best, score_mean=mean, score_std=std,
                )
                node.add_child(child)
                self.tree.backpropagate(child)
                if self.adaptive_enabled:
                    parent_population = RecipePopulation(
                        branch.manager.population, self.pop_size,
                        branch.manager.generation)
                    reward = self._adaptive_reward(parent_population, population)
                    self.adaptive_router.update(
                        recipe_id, adaptive_state, reward)
                self._write_branch_node(
                    child, branch_copy, recipe_id,
                    branch_copy.last_token_usage)
                children.append(child)
                print(
                    f"Generated PartEvo population node {child.population_node_id} "
                    f"recipe={recipe_id} generation={branch_copy.manager.generation} "
                    f"niches={len(branch_copy.manager.cluster_units)}",
                    flush=True,
                )

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(pending)) as executor:
            futures = [executor.submit(expand_one, recipe_id, recipe)
                       for recipe_id, recipe in pending]
            for future in futures:
                future.result()
        return children
