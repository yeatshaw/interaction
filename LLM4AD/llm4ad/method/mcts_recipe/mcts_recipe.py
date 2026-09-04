from __future__ import annotations

import copy
import json
import os
import base64
import concurrent.futures
import pickle
import random
import threading
from dataclasses import dataclass

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
                 node_batch_size=10, store_dir="recipe_mcts",
                 expand_fn=None, seed=None):
        recipes = validate_recipes(recipes)
        self.recipes = {
            k: (v if isinstance(v, RecipeConfig) else RecipeConfig(k, dict(v)))
            for k, v in recipes.items()
        }
        self.pop_size = int(pop_size)
        self.selection_num = int(selection_num)
        self.max_depth = int(max_depth)
        self.expand_fn = expand_fn
        # UNCERTAIN: the exact recipe definitions and EoH integration callback
        # are deliberately supplied by the task runner, not hard-coded here.
        self.tree = RecipeMCTS(self.recipes, exploration_constant, max_depth)
        self.store = RecipeStore(store_dir, node_batch_size)
        self._next_population_node_id = 1
        self._next_algorithm_id = 1
        self._checkpoint_index = max((
            int(entry.name[len("checkpoint_"):-5])
            for entry in os.scandir(self.store.directory)
            if entry.name.startswith("checkpoint_") and entry.name.endswith(".json")
            and entry.name[len("checkpoint_"):-5].isdigit()
        ), default=0)
        self._tree_lock = threading.RLock()
        self._best_samples = self.store.load_best_samples()
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
        self._total_sample_count = 0
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)

    def initialize(self, individuals):
        population = RecipePopulation(copy.deepcopy(individuals), self.pop_size)
        if not population.individuals:
            raise ValueError("initial population cannot be empty")
        self._assign_algorithm_ids(population.individuals)
        best = max(x.score for x in population.individuals)
        root_id = self._next_population_node_id
        self._next_population_node_id += 1
        root = RecipeNode(root_id, best, depth=0)
        self.tree.root = root
        self._record_best_candidates(population.individuals, root)
        self._write_node(root, population, None, [])
        # Persist the root once immediately. Subsequent nodes use batching.
        if self.store.flush() is not None:
            self._checkpoint_index += 1
            self._write_checkpoint()
        return root

    def _assign_algorithm_ids(self, individuals):
        for item in individuals:
            if getattr(item, "_recipe_algorithm_id", None) is None:
                item._recipe_algorithm_id = self._next_algorithm_id
                self._next_algorithm_id += 1

    def _record_best_candidates(self, individuals, population_node):
        """Append and persist each strict global-best improvement.

        This method is called only after an individual has been evaluated.  A
        candidate is compared with the process-wide ``_best_score``; when it
        improves that value, the record is appended and ``sample_best.json``
        is written immediately.  It is intentionally independent of node
        population batching.
        """
        for individual in individuals:
            self._total_sample_count += 1
            score = getattr(individual, "score", None)
            if score is None or score <= self._best_score:
                continue
            code = (individual.to_code_without_docstring()
                    if hasattr(individual, "to_code_without_docstring")
                    else str(individual))
            self._best_score = float(score)
            best_record = {
                "sample_order": self._total_sample_count,
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
            }
            self._best_samples.append(best_record)
            self.store.write_best_samples(self._best_samples)

    def _write_node(self, node, population, recipe_id, experiences):
        flushed = self.store.add(
            node.population_node_id, node.parent_population_node_id,
            node.depth, recipe_id, population.generation, population,
            experiences)
        if flushed is not None:
            self._checkpoint_index += 1
            self._write_checkpoint()

    def _load_node_state(self, node_id):
        from .resume import function_from_record

        record = self.store.load_node(node_id)
        functions = [function_from_record(item)
                     for item in record.get("algorithms", [])]
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
        population = RecipePopulation(functions, self.pop_size,
                                      record.get("generation", 0))
        return population, list(record.get("experiences", []))

    def expand_node(self, node, population, experiences=None):
        """Expand every recipe exactly once from the same parent snapshot."""
        if self.expand_fn is None:
            raise NotImplementedError(
                "Provide expand_fn implementing EoH reflection + operator scheduling.")
        pending_recipes = [
            (recipe_id, recipe)
            for recipe_id, recipe in self.recipes.items()
            if recipe_id not in node.expanded_recipe_ids
        ]
        if not pending_recipes:
            raise RuntimeError(
                f"Population node {node.population_node_id} is already fully expanded")
        children = []

        def expand_recipe(recipe_id, recipe):
            branch_experiences = copy.deepcopy(list(experiences or []))
            branch_population = population.clone()
            expander = (self.expand_fn[recipe_id]
                        if isinstance(self.expand_fn, dict) else self.expand_fn)
            if recipe.values.get("refineevo_experience", False):
                result = expander(
                    branch_population, recipe, self.selection_num,
                    self.pop_size, experiences=branch_experiences)
            else:
                result = expander(
                    branch_population, recipe, self.selection_num, self.pop_size)
            if isinstance(result, tuple):
                offspring, new_experiences = result
            else:
                offspring, new_experiences = result, []
            with self._tree_lock:
                if offspring:
                    self._assign_algorithm_ids(offspring)
                    child_population = branch_population.survival(offspring)
                else:
                    child_population = branch_population.inherited_next_generation()
                child_id = self._next_population_node_id
                self._next_population_node_id += 1
                child = RecipeNode(
                    child_id,
                    max(x.score for x in child_population.individuals),
                    depth=node.depth + 1, parent=node,
                    incoming_recipe_id=recipe_id)
                if offspring:
                    # Match EoH's sample-level bookkeeping: each evaluated
                    # individual is compared with the global best and any
                    # improvement is persisted immediately.  This is
                    # independent of population-node batch flushing.
                    for individual in offspring:
                        self._record_best_candidates([individual], child)
                node.add_child(child)
                # One completed recipe edge is one visit to its new child and
                # one additional visit propagated through every ancestor.
                self.tree.backpropagate(child)
                branch_experiences.extend(new_experiences)
                self._write_node(child, child_population, recipe_id,
                                 branch_experiences)
                children.append(child)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(pending_recipes)) as executor:
            futures = [executor.submit(expand_recipe, recipe_id, recipe)
                       for recipe_id, recipe in pending_recipes]
            for future in futures:
                future.result()
        return children

    def run(self, initial_individuals=None, checkpoint=None):
        if checkpoint is not None:
            self.restore(checkpoint)
        else:
            if initial_individuals is None:
                raise ValueError("initial_individuals is required for a new run")
            if self.store.next_file_start > 1:
                raise FileExistsError(
                    f"{self.store.directory} already contains population nodes; "
                    "use a new LLM4AD_LOG_DIR or set LLM4AD_CHECKPOINT")
            self.initialize(initial_individuals)
        while True:
            node = self.tree.select()
            if node.depth >= self.max_depth:
                break
            population, experiences = self._load_node_state(node.population_node_id)
            children = self.expand_node(node, population, experiences)
            if not children or any(child.depth >= self.max_depth for child in children):
                break
        self.store.flush()
        self._checkpoint_index += 1
        self._write_checkpoint()
        return self.tree.root

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
        if "exploration_constant" in state:
            self.tree.exploration_constant = float(state["exploration_constant"])
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
        self.restore_random_state(state)

    def _write_checkpoint(self):
        path = os.path.join(self.store.directory,
                            f"checkpoint_{self._checkpoint_index:06d}.json")
        nodes = []
        if self.tree.root is not None:
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
                })
                stack.extend(node.children)
        state = {
            # The counter always points to the next unused ID.  Keep both
            # fields for compatibility with older checkpoints.
            "last_population_node_id": max(0, self._next_population_node_id - 1),
            "next_population_node_id": self._next_population_node_id,
            "next_algorithm_id": self._next_algorithm_id,
            "total_sample_count": self._total_sample_count,
            "best_score": self._best_score,
            "max_depth": self.max_depth,
            "pop_size": self.pop_size,
            "selection_num": self.selection_num,
            "exploration_constant": self.tree.exploration_constant,
            "nodes": nodes,
            "recipes": {k: v.values for k, v in self.recipes.items()},
            # Pickled/base64 states preserve exact Python and NumPy RNG state;
            # they are local checkpoint data, not executable input.
            "python_random_state": self._encode_state(random.getstate()),
            "numpy_random_state": (
                self._encode_state(np.random.get_state()) if np is not None else None
            ),
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
