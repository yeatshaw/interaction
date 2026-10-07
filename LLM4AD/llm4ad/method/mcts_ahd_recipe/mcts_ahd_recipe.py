from __future__ import annotations

import copy
import base64
import concurrent.futures
import math
import os
import pickle
import random
import threading
import time
import traceback

from ..mcts_ahd import MCTS_AHD
from ..mcts_ahd.mcts import MCTS, MCTSNode
from ..mcts_ahd.population import Population
from ..mcts_ahd.prompt import MAEOHPrompt
from ...base import TextFunctionProgramConverter
from ..mcts_recipe.refineevo_experience import RefineEvoExperienceManager
from ..mcts_recipe.recipes import get_default_recipes
from .persistence import MCTSAHDRecipeStore


class MCTS_AHD_Recipe(MCTS_AHD):
    """MCTS-AHD variant with recipe-conditioned node expansion.

    The original MCTS-AHD implementation is untouched.  Tree selection,
    UCT, backpropagation and population survival remain inherited; only the
    prompt path is replaced with MCTS-AHD's own EoH-style prompt format plus
    a selected reflection recipe.  In batch mode, one selected parent node is
    expanded once per recipe, and every recipe executes the complete original
    operator round: e2 + 2*m1 + 2*m2 + s1.
    """

    def __init__(self, *args, info=None, recipe_configs=None,
                 recipe_temperature=1.0, recipe_batch_expansion=True,
                 recipes_per_node=5, embedding_model=None,
                 state_dir=None, node_batch_size=100,
                 expansion_workers=30, **kwargs):
        self._info = dict(info or {})
        if not self._info:
            raise ValueError("info is required for EoH-style recipe prompts")
        self._ensure_base_metadata(args, kwargs)

        batch_expansion_enabled = bool(recipe_batch_expansion)
        expansion_workers = int(expansion_workers)
        if expansion_workers < 1:
            raise ValueError("expansion_workers must be positive")

        super().__init__(*args, **kwargs)
        # One batch contains one task per operator invocation.  The base class
        # creates its evaluator pool before this subclass can see the batch
        # size, so num_evaluators=10 would otherwise throttle a 30-task batch
        # to ten concurrent evaluations. Rebuild only in batch mode; the
        # original non-batch execution keeps the base-class setting unchanged.
        if batch_expansion_enabled and self._num_evaluators < expansion_workers:
            self._evaluation_executor.shutdown(wait=True)
            self._num_evaluators = expansion_workers
            if self._multi_thread_or_process_eval == "thread":
                self._evaluation_executor = (
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=expansion_workers))
            else:
                self._evaluation_executor = (
                    concurrent.futures.ProcessPoolExecutor(
                        max_workers=expansion_workers))
        self._recipes = recipe_configs or get_default_recipes()
        self._temperature = float(recipe_temperature)
        if self._temperature <= 0:
            raise ValueError("recipe_temperature must be positive")
        self._recipe_batch_expansion = batch_expansion_enabled
        self._recipes_per_node = int(recipes_per_node)
        if self._recipes_per_node < 1:
            raise ValueError("recipes_per_node must be positive")
        if self._recipes_per_node > len(self._recipes):
            raise ValueError(
                "recipes_per_node cannot exceed the number of configured recipes")
        self._expansion_workers = expansion_workers
        self._recipe_gain = {name: 0.0 for name in self._recipes}
        self._recipe_count = {name: 0 for name in self._recipes}
        self._recipe_lock = threading.RLock()
        if state_dir is None:
            state_dir = os.path.join(
                getattr(self._profiler, "_log_dir", None)
                or "logs/mcts_ahd_recipe", "mcts_ahd_recipe_state")
        self._store = MCTSAHDRecipeStore(state_dir, node_batch_size)
        self._checkpoint_index = 0
        self._tree_checkpoint_pending = False
        self._tree_root_id = None
        self._pending_tree_updates = {}
        self._next_node_id = 1
        self._active_mcts = None
        self._resume_restored = False
        self._cumulative_token_usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "unknown_token_calls": 0,
        }
        self._convergence_records = []
        self._best_so_far = float("-inf")
        self._sample_order_lock = threading.RLock()
        self._sample_reservation_local = threading.local()
        self._embedding_model = embedding_model
        self._refineevo_manager = RefineEvoExperienceManager(
            reflector_llm=self._sampler.llm,
            embedding_model=embedding_model,
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

    def _merge_token_usage(self, usage):
        if not isinstance(usage, dict):
            return
        for field in (
                "calls", "prompt_tokens", "completion_tokens",
                "total_tokens", "unknown_token_calls"):
            self._cumulative_token_usage[field] += int(
                usage.get(field, 0) or 0)

    def _sample_order(self):
        return int(self._tot_sample_nums)

    def _sample_budget_exhausted(self):
        with self._sample_order_lock:
            return (self._max_sample_nums is not None
                    and self._tot_sample_nums >= self._max_sample_nums)

    def _on_evaluation_start(self):
        with self._sample_order_lock:
            if (self._max_sample_nums is not None
                    and self._tot_sample_nums >= self._max_sample_nums):
                return False
            self._tot_sample_nums += 1
            self._sample_reservation_local.sample_order = int(
                self._tot_sample_nums)
            self._sample_reservation_local.counted = True
            return True

    def _increment_sample_count(self):
        """The evaluation-start hook already counted this reserved sample."""
        if getattr(self._sample_reservation_local, "counted", False):
            self._sample_reservation_local.counted = False
            return
        return None

    def _record_convergence(self, function, sample_order):
        score = getattr(function, "score", None)
        try:
            numeric_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            numeric_score = None
        if numeric_score is not None and numeric_score > self._best_so_far:
            self._best_so_far = numeric_score
        self._convergence_records.append({
            "sample_order": int(sample_order),
            "score": numeric_score,
            "best_so_far": (self._best_so_far
                            if self._best_so_far != float("-inf") else None),
            "node_id": getattr(function, "_mcts_ahd_node_id", None),
            "recipe_id": getattr(function, "_recipe_id", None),
            "operator": getattr(function, "operator", None),
        })

    def _candidate_from_population(self, before_ids):
        candidates = list(self._population.next_gen_pop)
        candidates.extend(self._population.population)
        for candidate in reversed(candidates):
            if id(candidate) not in before_ids:
                return candidate
        return None

    def _sample_evaluate_register(self, prompt, func_only=False):
        if self._sample_budget_exhausted():
            return False
        self._sample_reservation_local.sample_order = None
        before_ids = {id(x) for x in self._population.population}
        before_ids.update(id(x) for x in self._population.next_gen_pop)
        usage = {}
        try:
            with self._sampler.llm.capture_token_usage() as usage:
                result = super()._sample_evaluate_register(
                    prompt, func_only=func_only)
        finally:
            self._sample_reservation_local.counted = False
        sample_order = getattr(self._sample_reservation_local,
                                "sample_order", None)
        self._sample_reservation_local.sample_order = None
        self._merge_token_usage(usage)
        candidate = (result if hasattr(result, "score")
                     else self._candidate_from_population(before_ids))
        if candidate is not None and sample_order is not None:
            candidate._recipe_token_usage = dict(usage)
            candidate._recipe_sample_order = sample_order
            self._record_convergence(
                candidate, candidate._recipe_sample_order)
        return result

    def _persist_node(self, node, refs=None, experiences=None):
        record = self._store.node_record(node, refs, experiences)
        if self._store.add(record) is not None:
            # The disk tree index is updated only after the complete parallel
            # expansion has committed, so checkpointing is deferred to run().
            self._tree_checkpoint_pending = True

    @staticmethod
    def _tree_metadata(mcts):
        if mcts is None or mcts.root is None:
            return []
        records = []
        stack = [mcts.root]
        while stack:
            node = stack.pop()
            records.append({
                "node_id": int(node.node_id),
                "parent_node_id": (
                    int(node.parent.node_id)
                    if node.parent is not None else None
                ),
                "depth": int(node.depth),
                "Q": node.Q,
                "visits": int(node.visits),
                "children_node_ids": [int(child.node_id)
                                       for child in node.children],
            })
            stack.extend(node.children)
        return records

    def _population_state(self):
        return {
            "population": [
                self._store._algorithm_record(item)
                for item in self._population.population
            ],
            "next_gen_pop": [
                self._store._algorithm_record(item)
                for item in self._population.next_gen_pop
            ],
            "generation": int(self._population.generation),
        }

    def _write_checkpoint(self):
        state = {
            "next_node_id": int(self._next_node_id),
            "total_sample_nums": int(self._tot_sample_nums),
            "population": self._population_state(),
            # The full tree is never retained in the checkpointing object.
            # Search metadata is persisted separately by MCTSAHDRecipeStore.
            "tree_state_file": "tree_state.json",
            "recipe_gain": dict(self._recipe_gain),
            "recipe_count": dict(self._recipe_count),
            "cumulative_token_usage": dict(self._cumulative_token_usage),
            "convergence_records": list(self._convergence_records),
            "best_so_far": (self._best_so_far
                            if self._best_so_far != float("-inf") else None),
            "recipe_batch_expansion": self._recipe_batch_expansion,
            "recipes_per_node": self._recipes_per_node,
            "expansion_workers": self._expansion_workers,
            "python_random_state": base64.b64encode(
                pickle.dumps(random.getstate(), protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii"),
        }
        if self._active_mcts is not None and self._active_mcts.root is not None:
            state["root_Q"] = self._active_mcts.root.Q
        self._store.write_checkpoint(self._checkpoint_index, state)

    def _persist_initial_tree(self, mcts):
        mcts.root.node_id = 0
        for child in mcts.root.children:
            child.node_id = self._next_node_id
            self._next_node_id += 1
            child.individual._recipe_algorithm_id = child.node_id
            child.individual._mcts_ahd_node_id = child.node_id
            child.individual._eoh_parent_ids = ()
        # Assign every child ID before serializing the root's child list.
        self._persist_node(mcts.root)
        for child in mcts.root.children:
            self._persist_node(child, refs=[])
        self._store.flush()
        self._write_tree_state_from_working_tree(mcts)
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self._tree_checkpoint_pending = False
        self._active_mcts = None

    def _restore_population_function(self, record):
        function = self._store.function_from_record(record)
        return function

    def _attach_parent_functions(self, functions):
        from .persistence import MCTSAHDRecipeStore

        parent_ids = {
            parent_id
            for function in functions
            for parent_id in getattr(function, "_eoh_parent_ids", ())
            if parent_id is not None
        }
        records = self._store.load_algorithm_records(parent_ids)
        parents = {
            int(parent_id): self._restore_population_function(record)
            for parent_id, record in records.items()
        }
        for function in functions:
            function._recipe_parent_functions = tuple(
                parents[parent_id]
                for parent_id in getattr(function, "_eoh_parent_ids", ())
                if int(parent_id) in parents
            )

    @staticmethod
    def _working_tree_record(node):
        return {
            "node_id": int(node.node_id),
            "parent_node_id": (
                int(node.parent.node_id) if node.parent is not None else None),
            "depth": int(node.depth),
            "Q": float(node.Q),
            "visits": int(node.visits),
            "children_node_ids": [int(child.node_id)
                                   for child in node.children],
        }

    def _write_tree_state_from_working_tree(self, mcts):
        payload = self._tree_payload()
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        if mcts is None or mcts.root is None:
            return
        stack = [mcts.root]
        while stack:
            node = stack.pop()
            item = self._working_tree_record(node)
            old = by_id.get(int(node.node_id), {})
            if not node.children and old.get("children_node_ids"):
                item["children_node_ids"] = list(old["children_node_ids"])
            by_id[int(node.node_id)] = item
            stack.extend(node.children)
        changed_ids = set()
        stack = [mcts.root]
        while stack:
            node = stack.pop()
            changed_ids.add(int(node.node_id))
            stack.extend(node.children)
        self._tree_root_id = int(mcts.root.node_id)
        for node_id in changed_ids:
            if node_id in by_id:
                self._pending_tree_updates[node_id] = dict(by_id[node_id])

    def _tree_payload(self):
        payload = self._store.load_tree_state()
        root_id = (self._tree_root_id
                   if self._tree_root_id is not None
                   else payload.get("root_id"))
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        for node_id, item in self._pending_tree_updates.items():
            by_id[int(node_id)] = dict(item)
        return {"root_id": root_id, "nodes": list(by_id.values())}

    def _commit_pending_tree_state(self):
        if not self._pending_tree_updates:
            return
        payload = self._store.load_tree_state()
        by_id = {}
        for item in payload.get("nodes", []):
            try:
                by_id[int(item["node_id"])] = dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        for node_id, item in self._pending_tree_updates.items():
            by_id[int(node_id)] = dict(item)
        self._store.write_tree_state(
            self._tree_root_id if self._tree_root_id is not None
            else payload.get("root_id"),
            list(by_id.values()),
        )
        self._pending_tree_updates = {}

    def _load_working_tree(self):
        """Load only the root, root children, and selected path into RAM."""
        payload = self._tree_payload()
        metadata = {}
        for item in payload.get("nodes", []):
            try:
                metadata[int(item["node_id"])] = item
            except (KeyError, TypeError, ValueError):
                continue
        root_id = payload.get("root_id")
        if root_id is None or int(root_id) not in metadata:
            return None, None
        root_id = int(root_id)
        children_by_parent = {}
        for item in metadata.values():
            parent_id = item.get("parent_node_id")
            if parent_id is not None:
                children_by_parent.setdefault(int(parent_id), []).append(item)

        mcts = MCTS("Root", self.alpha, self.lambda_0)
        mcts.q_min = min(
            [0.0] + [float(item.get("Q", 0.0)) for item in metadata.values()])
        mcts.q_max = max(
            [-10000.0] + [float(item.get("Q", 0.0)) for item in metadata.values()])
        mcts.rank_list = sorted({float(item.get("Q", 0.0))
                                 for item in metadata.values()})
        root = MCTSNode("Root", "Root", 0, depth=0)
        root.node_id = root_id
        root.Q = float(metadata[root_id].get("Q", 0.0))
        root.visits = int(metadata[root_id].get("visits", 1))
        mcts.root = root

        def load_node(item, parent):
            record = self._store.load_node(int(item["node_id"]))
            algorithm_record = record.get("algorithm")
            if not algorithm_record:
                return None
            function = self._restore_population_function(algorithm_record)
            self._attach_parent_functions([function])
            node = MCTSNode(
                function.algorithm, str(function), -1 * function.score,
                individual=function, parent=parent,
                depth=int(item.get("depth", 0)),
                visit=int(item.get("visits", 1)),
                Q=float(item.get("Q", function.score)), raw_info=function)
            node.node_id = int(item["node_id"])
            node._refineevo_experiences = list(
                record.get("experiences", []) or [])
            return node

        def attach_children(parent, items):
            result = []
            for item in sorted(items, key=lambda value: int(value["node_id"])):
                child = load_node(item, parent)
                if child is None:
                    continue
                parent.add_child(child)
                if parent is root:
                    child.subtree = [child]
                result.append(child)
            return result

        current = root
        current_children = attach_children(root, children_by_parent.get(root_id, []))
        while current_children and current.depth < mcts.max_depth:
            eval_remain = max(
                1 - self._tot_sample_nums / self._max_sample_nums, 0)
            selected = max(
                current_children,
                key=lambda child: mcts.uct(child, eval_remain),
            )
            if int(current.visits ** mcts.alpha) > len(current_children):
                option = "e1" if current is root else "e2"
                before = len(current.children)
                self.expand(mcts, current.children, current, option)
                if len(current.children) > before:
                    self._write_tree_state_from_working_tree(mcts)
                    self._tree_checkpoint_pending = True
            current = selected
            current_children = attach_children(
                current, children_by_parent.get(int(current.node_id), []))
        return mcts, current

    def _restore_from_checkpoint(self):
        state, path = self._store.latest_checkpoint()
        if state is None:
            return None
        tree_state = self._store.load_tree_state()
        if tree_state.get("root_id") is None and state.get("tree"):
            # Compatibility with checkpoints created before the disk index
            # was introduced. The conversion is written once, then all later
            # iterations use the compact on-disk index.
            converted = []
            for item in state.get("tree", []):
                converted.append({
                    "node_id": item.get("node_id"),
                    "parent_node_id": item.get("parent_node_id"),
                    "depth": item.get("depth", 0),
                    "Q": item.get("Q", 0.0),
                    "visits": item.get("visits", 1),
                    "children_node_ids": item.get("children_node_ids", []),
                })
            if converted:
                root_id = next((item["node_id"] for item in converted
                                if item.get("parent_node_id") is None), 0)
                self._store.write_tree_state(root_id, converted)
        self._tree_root_id = self._store.load_tree_state().get("root_id")
        self._pending_tree_updates = {}

        population_state = state.get("population", {})
        population = Population(
            init_pop_size=self._init_pop_size,
            pop_size=self._pop_size,
            generation=int(population_state.get("generation", 0)),
        )
        population._population = [
            self._restore_population_function(item)
            for item in population_state.get("population", [])
        ]
        population._next_gen_pop = [
            self._restore_population_function(item)
            for item in population_state.get("next_gen_pop", [])
        ]
        self._population = population
        self._attach_parent_functions(
            list(population.population) + list(population.next_gen_pop))
        self._next_node_id = int(state.get("next_node_id", 1))
        self._tot_sample_nums = int(state.get("total_sample_nums", 0))
        self._recipe_gain.update(state.get("recipe_gain", {}))
        self._recipe_count.update(state.get("recipe_count", {}))
        self._cumulative_token_usage.update(
            state.get("cumulative_token_usage", {}))
        self._convergence_records = list(state.get("convergence_records", []))
        best = state.get("best_so_far")
        self._best_so_far = float(best) if best is not None else float("-inf")
        encoded_random_state = state.get("python_random_state")
        if encoded_random_state:
            random.setstate(pickle.loads(base64.b64decode(
                encoded_random_state.encode("ascii"))))
        checkpoint_indices = []
        for name in os.listdir(self._store.directory):
            if name.startswith("checkpoint_") and name.endswith(".json"):
                value = name[len("checkpoint_"):-5]
                if value.isdigit():
                    checkpoint_indices.append(int(value))
        self._checkpoint_index = max(
            [self._checkpoint_index] + checkpoint_indices)
        if self._profiler is not None:
            self._profiler._num_samples = self._tot_sample_nums
        self._active_mcts = None
        self._resume_restored = True
        self._tree_checkpoint_pending = False
        return None

    def _finish_persistence(self):
        self._store.flush()
        self._commit_pending_tree_state()
        self._checkpoint_index += 1
        self._write_checkpoint()
        self._store.write_convergence(self._convergence_records)
        self._store.write_convergence_plot(self._convergence_records)
        self._store.write_token_usage(self._cumulative_token_usage)
        print(
            "Total LLM tokens: "
            f"input={self._cumulative_token_usage['prompt_tokens']}, "
            f"output={self._cumulative_token_usage['completion_tokens']}, "
            f"total={self._cumulative_token_usage['total_tokens']}",
            flush=True,
        )

    def _choose_recipe(self, available_names=None):
        with self._recipe_lock:
            names = list(available_names or self._recipes)
            values = [self._recipe_gain[name] / self._temperature for name in names]
        shift = max(values)
        weights = [math.exp(value - shift) for value in values]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        index = random.choices(range(len(names)), weights=probabilities, k=1)[0]
        return names[index], probabilities[index]

    def _recipe_probability(self, recipe_id, available_names=None):
        with self._recipe_lock:
            names = list(available_names or self._recipes)
            values = [self._recipe_gain[name] / self._temperature for name in names]
        shift = max(values)
        weights = [math.exp(value - shift) for value in values]
        total = sum(weights)
        return weights[names.index(recipe_id)] / total

    def _choose_recipe_batch(self):
        """Select distinct recipes for one parent-node expansion batch."""
        available = list(self._recipes)
        selected = []
        for _ in range(self._recipes_per_node):
            recipe_id, probability = self._choose_recipe(available)
            selected.append((recipe_id, probability))
            available.remove(recipe_id)
        return selected

    @staticmethod
    def _lineage_parents(refs):
        return [list(getattr(ref, "_recipe_parent_functions", ())) for ref in refs]

    @staticmethod
    def _reference_copy(function):
        """Copy only the prompt-facing Function shell.

        A recursive deepcopy here duplicates the complete parent lineage for
        every parallel task and every newly generated algorithm.  The lineage
        itself is immutable for prompt construction and is restored from
        ``_eoh_parent_ids`` on checkpoint resume, so a shallow Function copy
        is sufficient and avoids quadratic/exponential memory growth.
        """
        shell = copy.copy(function)
        # Preserve one prompt-facing parent level without retaining the
        # complete recursive lineage through every shell.
        parents = []
        for parent in getattr(function, "_recipe_parent_functions", ()):
            parent_shell = copy.copy(parent)
            parent_shell._recipe_parent_functions = ()
            parents.append(parent_shell)
        shell._recipe_parent_functions = tuple(parents)
        return shell

    @staticmethod
    def _compact_subtree_index(mcts):
        """Keep only the representative needed by the root e1 prompt.

        ``subtree`` is used only to obtain the first representative under
        each root child.  MCTS selection still uses the full parent/children
        tree, which remains untouched.
        """
        if mcts is None or mcts.root is None:
            return
        for root_child in mcts.root.children:
            root_child.subtree = [root_child]
            stack = list(root_child.children)
            while stack:
                node = stack.pop()
                node.subtree = []
                stack.extend(node.children)

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

    def _reflection(self, refs, recipe_id, population=None):
        config = self._recipes[recipe_id]
        prompt = MAEOHPrompt.get_prompt_reflection(
            refs=list(refs), parents=self._lineage_parents(refs), info=self._info,
            parent_info_flag=config.get("parent_info", False),
            best_worst_flag=config.get("best_worst", False),
            fitness_flag=config.get("fitness", 0),
            avg_fitness_flag=config.get("avg_fitness", False),
            check_reflection_flag=config.get("check_guidance", False),
            population=population if population is not None else self._population,
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

    def _refineevo_reflection(self, refs, operator, cur_node, recipe_id,
                              manager=None, parent_experiences=None):
        config = self._recipes[recipe_id]
        if parent_experiences is None:
            parent_experiences = copy.deepcopy(
                getattr(cur_node, "_refineevo_experiences", []) or []
            )
        else:
            parent_experiences = copy.deepcopy(parent_experiences)
        manager = manager or self._refineevo_manager
        manager.top_k = int(config.get("experience_top_k", 3))
        manager_operator = self._refineevo_operator(operator)
        manager.begin_node(parent_experiences)
        try:
            retrieved = manager.retrieve(
                refs, manager_operator, parent_experiences)
            fresh = manager.distill(refs, manager_operator)
        finally:
            manager.end_node()
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
        func._recipe_id = recipe_id
        func.recipe_probability = probability
        func.operator = operator
        func._eoh_generation_suggestion = getattr(func, "_recipe_suggestion", None)
        baseline = max((float(ref.score) for ref in refs if ref.score is not None),
                       default=None)
        gain = max(0.0, float(func.score) - baseline) if baseline is not None else 0.0
        with self._recipe_lock:
            self._recipe_gain[recipe_id] += gain
            self._recipe_count[recipe_id] += 1
        # Parent IDs are persisted as the durable lineage.  Keep lightweight
        # prompt references in memory instead of recursively copying parents.
        func._recipe_parent_functions = tuple(refs)
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
        func._refineevo_experiences = experiences
        func._recipe_algorithm_id = self._next_node_id
        func._mcts_ahd_node_id = self._next_node_id
        func._eoh_parent_ids = tuple(
            getattr(ref, "_recipe_algorithm_id", None) for ref in refs)
        node._refineevo_experiences = experiences
        node.node_id = self._next_node_id
        self._next_node_id += 1
        if operator == "e1":
            node.subtree.append(node)
        cur_node.add_child(node)
        mcts.backpropagate(node)
        node_set.append(node)
        self._persist_node(node, refs, experiences)
        return node_set

    def expand(self, mcts, node_set, cur_node, option, recipe_id=None,
               recipe_probability=None):
        """Expand one MCTS action using a fixed or newly selected recipe."""
        if not self._continue_loop():
            return node_set
        if option == "e1":
            refs = [self._reference_copy(child.subtree[0].individual)
                    for child in mcts.root.children if child.subtree]
            refs = refs or [self._reference_copy(cur_node.individual)]
        elif option == "e2":
            other = self._population.selection()
            refs = [self._reference_copy(other),
                    self._reference_copy(cur_node.individual)]
        else:
            refs = [self._reference_copy(cur_node.individual)]

        if recipe_id is None:
            recipe_id, probability = self._choose_recipe()
        else:
            if recipe_id not in self._recipes:
                raise ValueError(f"Unknown recipe: {recipe_id}")
            probability = (
                self._recipe_probability(recipe_id)
                if recipe_probability is None else recipe_probability
            )
        try:
            config = self._recipes[recipe_id]
            parent_experiences = None
            retrieved_experiences = None
            fresh_experiences = None
            suggestion_title = None
            with self._sampler.llm.capture_token_usage() as reflection_usage:
                if config.get("refineevo_experience", False):
                    suggestion, parent_experiences, retrieved_experiences, fresh_experiences = (
                        self._refineevo_reflection(refs, option, cur_node, recipe_id)
                    )
                    suggestion_title = "These are successful and failed design experiences retrieved from previous attempts:"
                else:
                    suggestion = self._reflection(refs, recipe_id)
            self._merge_token_usage(reflection_usage)
            prompt = self._eoh_operator_prompt(
                option, refs, suggestion, suggestion_title)
            func = self._sample_evaluate_register(prompt, func_only=True)
            if func is not False and func is not None:
                func._recipe_suggestion = suggestion
                func._recipe_token_usage = self._sampler.llm.merge_token_usage(
                    getattr(func, "_recipe_token_usage", None),
                    reflection_usage,
                )
            return self._register_expanded_function(
                mcts, node_set, cur_node, func, option, refs,
                recipe_id, probability, parent_experiences,
                retrieved_experiences, fresh_experiences)
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return node_set

    def _parallel_expand_task(self, task):
        """Run reflection, generation and evaluation for one fixed task."""
        sample_order = None

        recipe_id = task["recipe_id"]
        option = task["operator"]
        refs = task["refs"]
        parent_experiences = task["parent_experiences"]
        population_snapshot = task["population"]
        config = self._recipes[recipe_id]
        sample_start = time.time()
        try:
            with self._sampler.llm.capture_token_usage() as usage:
                if config.get("refineevo_experience", False):
                    manager = RefineEvoExperienceManager(
                        reflector_llm=self._sampler.llm,
                        embedding_model=self._embedding_model,
                        top_k=int(config.get("experience_top_k", 3)),
                    )
                    suggestion, parent_experiences, retrieved, fresh = (
                        self._refineevo_reflection(
                            refs, option, task["parent_node"], recipe_id,
                            manager=manager,
                            parent_experiences=parent_experiences,
                        )
                    )
                    suggestion_title = (
                        "These are successful and failed design experiences "
                        "retrieved from previous attempts:"
                    )
                else:
                    suggestion = self._reflection(
                        refs, recipe_id, population=population_snapshot)
                    parent_experiences = None
                    retrieved = None
                    fresh = None
                    suggestion_title = None

                prompt = self._eoh_operator_prompt(
                    option, refs, suggestion, suggestion_title)
                thought, function = self._sampler.get_thought_and_function(
                    self._task_description_str, prompt)
                if thought is None or function is None:
                    return {
                        "sample_order": sample_order,
                        "usage": dict(usage),
                        "function": None,
                    }
                program = TextFunctionProgramConverter.function_to_program(
                    function, self._template_program)
                if program is None:
                    return {
                        "sample_order": None,
                        "usage": dict(usage),
                        "function": None,
                    }
                if not self._on_evaluation_start():
                    return {
                        "sample_order": None,
                        "usage": dict(usage),
                        "function": None,
                    }
                sample_order = self._sample_reservation_local.sample_order
                score, eval_time = self._evaluation_executor.submit(
                    self._evaluator.evaluate_program_record_time,
                    program,
                ).result()

            function.score = score
            function.evaluate_time = eval_time
            function.algorithm = thought
            function.sample_time = time.time() - sample_start
            function._recipe_token_usage = dict(usage)
            function._recipe_sample_order = sample_order
            function._recipe_suggestion = suggestion
            return {
                "sample_order": sample_order,
                "usage": dict(usage),
                "function": function,
                "program": str(program),
                "operator": option,
                "recipe_id": recipe_id,
                "recipe_probability": task["recipe_probability"],
                "refs": refs,
                "parent_experiences": parent_experiences,
                "retrieved_experiences": retrieved,
                "fresh_experiences": fresh,
            }
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return {
                "sample_order": sample_order,
                "usage": {},
                "function": None,
            }
        finally:
            self._sample_reservation_local.counted = False

    def _build_parallel_expansion_tasks(self, mcts, cur_node):
        population_snapshot = Population(
            init_pop_size=self._init_pop_size,
            pop_size=self._pop_size,
            generation=self._population.generation,
            # Selection and reflection only read this snapshot.  Sharing the
            # Function shells avoids copying every stored parent lineage once
            # per expansion batch.
            pop=list(self._population.population),
        )
        parent_experiences = copy.deepcopy(
            getattr(cur_node, "_refineevo_experiences", []) or [])
        tasks = []
        operator_round = ("e2", "m1", "m1", "m2", "m2", "s1")
        for recipe_id, probability in self._choose_recipe_batch():
            for operator in operator_round:
                if operator == "e2":
                    other = population_snapshot.selection(
                        population_snapshot.population)
                    refs = [self._reference_copy(other),
                            self._reference_copy(cur_node.individual)]
                else:
                    refs = [self._reference_copy(cur_node.individual)]
                tasks.append({
                    "recipe_id": recipe_id,
                    "recipe_probability": probability,
                    "operator": operator,
                    "refs": refs,
                    "parent_experiences": copy.deepcopy(parent_experiences),
                    "parent_node": cur_node,
                    "population": population_snapshot,
                })
        return tasks

    def _commit_parallel_results(self, mcts, cur_node, results):
        """Commit results only after every parallel task has finished."""
        node_set = []
        for result in sorted(
                (item for item in results if item is not None),
                key=lambda item: int(item.get("sample_order") or 0)):
            self._merge_token_usage(result.get("usage"))
            function = result.get("function")
            if function is None:
                continue
            if self._profiler is not None:
                self._profiler._num_samples = int(result["sample_order"]) - 1
                self._profiler.register_function(
                    function, program=result.get("program", ""))
            before_count = len(node_set)
            node_set = self._register_expanded_function(
                mcts,
                node_set,
                cur_node,
                function,
                result["operator"],
                result["refs"],
                result["recipe_id"],
                result["recipe_probability"],
                result.get("parent_experiences"),
                result.get("retrieved_experiences"),
                result.get("fresh_experiences"),
            )
            if len(node_set) > before_count:
                self._record_convergence(
                    function, int(result["sample_order"]))
        return node_set

    def _expand_node_recipe_batch(self, mcts, cur_node):
        """Expand one selected node with every recipe in parallel.

        Each worker performs reflection, LLM generation and evaluation. Tree
        mutation and persistence happen only after all workers complete.
        """
        if self._sample_budget_exhausted():
            return []
        tasks = self._build_parallel_expansion_tasks(mcts, cur_node)
        print(
            f"Parallel recipe batch: {len(tasks)} tasks, "
            f"llm_workers={min(self._expansion_workers, len(tasks))}, "
            f"evaluation_workers={getattr(self, '_num_evaluators', None)}, "
            f"parent depth={cur_node.depth}",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._expansion_workers, len(tasks))) as executor:
            results = list(executor.map(self._parallel_expand_task, tasks))
        node_set = self._commit_parallel_results(mcts, cur_node, results)
        self._compact_subtree_index(mcts)
        return node_set

    def run(self):
        """Run MCTS-AHD with optional recipe-level expansion batches."""
        if not self._recipe_batch_expansion:
            return super().run()

        self._restore_from_checkpoint() if self._resume_mode else None
        if not self._resume_restored:
            mcts = MCTS("Root", self.alpha, self.lambda_0)
            self._active_mcts = mcts

            self._init_one_solution()
            self._population.survival()
            self._iteratively_init_population_root()

            for indiv in self._population.population:
                now_node = MCTSNode(
                    indiv.algorithm, str(indiv), -1 * indiv.score,
                    individual=indiv, parent=mcts.root, depth=1, visit=1,
                    Q=indiv.score, raw_info=indiv,
                )
                mcts.root.add_child(now_node)
                mcts.backpropagate(now_node)
                now_node.subtree.append(now_node)
            self._persist_initial_tree(mcts)

        if len(self._population) < self._selection_num:
            print(
                f"The search is terminated since MCTS_AHD unable to obtain "
                f"{self._selection_num} feasible algorithms during initialization. "
                f"Please increase the `initial_sample_nums_max` argument "
                f"(currently {self._initial_sample_nums_max})."
            )
            self._finish_persistence()
            return

        while self._continue_loop():
            mcts, cur_node = self._load_working_tree()
            if mcts is None or cur_node is None:
                break
            self._active_mcts = mcts
            self._expand_node_recipe_batch(mcts, cur_node)
            self._write_tree_state_from_working_tree(mcts)
            if self._tree_checkpoint_pending:
                self._store.flush()
                self._commit_pending_tree_state()
                self._checkpoint_index += 1
                self._write_checkpoint()
                self._tree_checkpoint_pending = False
            self._population.survival()
            del cur_node, mcts
            self._active_mcts = None

        self._finish_persistence()
        if self._profiler is not None:
            self._profiler.finish()
        self._sampler.llm.close()

    def _iteratively_init_population_root(self):
        while (len(self._population.population) < self._init_pop_size
               and self._continue_loop()):
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
        while (len(self._population.next_gen_pop) == 0
               and self._continue_loop()):
            try:
                prompt = MAEOHPrompt.get_prompt_i1(self._info)
                self._sample_evaluate_register(prompt)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue
