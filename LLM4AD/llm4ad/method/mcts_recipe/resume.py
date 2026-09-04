from __future__ import annotations

import json
from pathlib import Path

from .mcts import RecipeNode
from .population import RecipePopulation
from ...base import TextFunctionProgramConverter


def load_checkpoint(path):
    """Load and validate minimal MCTS metadata.

    Reconstructing Functions from serialized code remains task-specific.
    """
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    required = {"last_population_node_id", "next_algorithm_id", "nodes",
                "pop_size", "selection_num", "max_depth", "recipes"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    return state


def restore_random_state(state):
    """Restore Python/NumPy RNG state saved by :class:`MCTSRecipe`."""
    import base64
    import pickle
    import random

    if state.get("python_random_state"):
        random.setstate(pickle.loads(base64.b64decode(
            state["python_random_state"].encode("ascii"))))
    if state.get("numpy_random_state"):
        import numpy as np
        np.random.set_state(pickle.loads(base64.b64decode(
            state["numpy_random_state"].encode("ascii"))))


def function_from_record(record, template_program=None):
    """Recreate an LLM4AD Function and restore persisted metadata."""
    function = TextFunctionProgramConverter.text_to_function(record["code"])
    if function is None:
        raise ValueError(f"Invalid algorithm code for id {record.get('algorithm_id')}")
    function.score = record.get("score")
    function.algorithm = record.get("thought", "")
    function._recipe_algorithm_id = record.get("algorithm_id")
    function._eoh_parent_ids = tuple(record.get("parent_algorithm_ids", ()))
    function._eoh_generation_suggestion = record.get("suggestion")
    function._eoh_experience = record.get("experience")
    function.operator = record.get("operator")
    function.evaluate_time = record.get("evaluate_time")
    function.sample_time = record.get("sample_time")
    function._recipe_id = record.get("recipe_id")
    return function


def load_population_nodes(directory, template_program=None):
    """Load all persisted population snapshots, indexed by node ID."""
    populations = {}
    for path in sorted(Path(directory).glob("population_nodes_*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        for node in payload.get("nodes", []):
            funcs = [function_from_record(x, template_program)
                     for x in node.get("algorithms", [])]
            populations[int(node["population_node_id"])] = RecipePopulation(
                funcs, int(node.get("population_size", len(funcs))),
                int(node.get("generation", 0)))
    return populations


def restore_tree(checkpoint):
    """Rebuild the lightweight MCTS tree from checkpoint node metadata."""
    records = {int(x["population_node_id"]): x for x in checkpoint.get("nodes", [])}
    nodes = {}
    for node_id, record in records.items():
        nodes[node_id] = RecipeNode(node_id, record["Q"], record["depth"],
                                    visits=record.get("visits", 1),
                                    incoming_recipe_id=record.get("incoming_recipe_id"))
    root = None
    for node_id, record in records.items():
        parent_id = record.get("parent_population_node_id")
        if parent_id is None:
            root = nodes[node_id]
        else:
            parent = nodes[int(parent_id)]
            nodes[node_id].parent = parent
            nodes[node_id].parent_population_node_id = int(parent_id)
            parent.children.append(nodes[node_id])
            recipe_id = record.get("incoming_recipe_id")
            if recipe_id is not None:
                parent.expanded_recipe_ids.add(recipe_id)
        nodes[node_id].expanded_recipe_ids.update(
            record.get("expanded_recipe_ids", ()))
    return root, nodes
