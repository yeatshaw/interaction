"""Offline checks against the original in-memory RecipeMCTS behavior.

Load the tree and runner functions without importing optional LLM/evaluator
packages so these checks require only Python's standard library.
"""

import ast
import concurrent.futures
import copy
import importlib.util
import math
import random
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TREE_DIR = ROOT / "llm4ad/method/mcts_recipe"
spec = importlib.util.spec_from_file_location("recipe_tree", TREE_DIR / "mcts.py")
tree_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tree_module)
RecipeNode = tree_module.RecipeNode
RecipeMCTS = tree_module.RecipeMCTS


def load_function(path, name, namespace, class_name=None):
    module = ast.parse(path.read_text(encoding="utf-8"))
    body = module.body
    if class_name:
        body = next(item.body for item in body
                    if isinstance(item, ast.ClassDef) and item.name == class_name)
    function = next(item for item in body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
    # The isolated module already supplies the actual RecipeNode class.
    function.body = [item for item in function.body
                     if not isinstance(item, ast.ImportFrom)]
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


class LegacyFlowTests(unittest.TestCase):
    def test_disk_search_matches_in_memory_search_and_backpropagation(self):
        namespace = {"math": math, "RecipeNode": RecipeNode}
        select = load_function(TREE_DIR / "mcts_recipe.py", "_select_disk_tree_node",
                               namespace, "MCTSRecipe")
        transition = load_function(TREE_DIR / "mcts_recipe.py",
                                   "_write_disk_tree_transition", namespace,
                                   "MCTSRecipe")
        for balance in (0.0, 0.4):
            with self.subTest(depth_balance_weight=balance):
                rng = random.Random(17)
                recipes = dict.fromkeys("abcde")
                tree = RecipeMCTS(recipes, exploration_constant=0.2,
                                  max_depth=200, depth_balance_weight=balance)
                root = RecipeNode(1, -10)
                tree.root = root
                nodes = {1: root}
                records = {1: {
                    "population_node_id": 1, "parent_population_node_id": None,
                    "depth": 0, "Q": -10, "visits": 1,
                    "expanded_recipe_ids": [], "children_population_node_ids": [],
                }}
                disk = SimpleNamespace(recipes=recipes, tree=tree, max_depth=200,
                                       depth_balance_weight=balance,
                                       allow_recipe_repeats=False)
                disk._disk_tree_records = lambda: (1, copy.deepcopy(records))

                def save(root_id, updated, changed_ids):
                    records.update({key: copy.deepcopy(updated[key])
                                    for key in changed_ids})

                disk._set_pending_tree_state = save
                next_id = 2
                for step in range(100):
                    old_parent = tree.select()
                    new_parent = select(disk)
                    self.assertEqual(new_parent.population_node_id,
                                     old_parent.population_node_id, step)
                    # IDs are reserved before parallel execution; completion
                    # order can differ from both ID and recipe order.
                    edges = list(zip(recipes, range(next_id, next_id + 5)))
                    next_id += 5
                    rng.shuffle(edges)
                    children = []
                    for recipe_id, node_id in edges:
                        q = max(old_parent.Q, rng.choice((-10, -10, -9, -8)))
                        old_child = RecipeNode(node_id, q, old_parent.depth + 1,
                                               old_parent, recipe_id)
                        old_parent.add_child(old_child)
                        tree.backpropagate(old_child)
                        nodes[node_id] = old_child
                        new_child = RecipeNode(node_id, q, new_parent.depth + 1,
                                               new_parent, recipe_id)
                        new_parent.add_child(new_child)
                        tree.backpropagate(new_child)
                        children.append(new_child)
                    transition(disk, new_parent, children)
                    for node_id, node in nodes.items():
                        self.assertEqual(records[node_id]["visits"], node.visits,
                                         (step, node_id))
                        self.assertEqual(records[node_id]["Q"], node.Q)
                        self.assertEqual(
                            records[node_id].get("children_population_node_ids", []),
                            [child.population_node_id for child in node.children])
                    # Physical record order must not affect the next selection.
                    items = list(records.items())
                    rng.shuffle(items)
                    records = dict(items)

    def test_refineevo_initialization_counts_failed_generation_attempts(self):
        calls = []
        reservations = []
        evaluations = []

        def sample(*args, **kwargs):
            calls.append(len(reservations))
            if len(calls) < 3:
                return None, None
            return "algorithm", SimpleNamespace()

        def reserve():
            reservations.append(len(reservations) + 1)
            return reservations[-1]

        namespace = {
            "SecureEvaluator": lambda *a, **kw: SimpleNamespace(
                evaluate_program_record_time=lambda program: (1.234567, 0.1)),
            "_refineevo_initial_prompt": lambda *a: "prompt",
            "_token_usage_helpers": lambda llm: (
                lambda: nullcontext({}), lambda usage: None, lambda: {}),
            "sample_initial_algorithm": sample,
            "TextFunctionProgramConverter": SimpleNamespace(
                function_to_program=lambda function, template: function),
            "NodeEmbeddingError": type("NodeEmbeddingError", (Exception,), {}),
            "SampleBudgetExhausted": type("SampleBudgetExhausted", (Exception,), {}),
            "np": SimpleNamespace(round=round), "time": time, "math": math,
            "concurrent": concurrent, "json": __import__("json"),
            "print": lambda *a, **kw: None,
        }
        initialize = load_function(ROOT / "example/tasks/mcts_recipe_runner.py",
                                   "_initial_population", namespace)
        population, _ = initialize(
            None, None, {"template_program": "template", "template_module": "tsp"},
            pop_size=1, selection_num=1, num_samplers=1, init_pop_size=1,
            reserve_sample_order=reserve, on_evaluated=evaluations.append)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(population[0]._recipe_sample_order, 3)
        self.assertEqual(population[0].score, 1.23457)
        self.assertEqual(evaluations, population)

        # The shared helper must retain PartEvo's evaluation-only admission.
        calls.clear()
        reservations.clear()
        admissions = []

        def admit():
            admissions.append(1)
            return len(admissions)

        population, _ = initialize(
            None, None, {"template_program": "template", "template_module": "tsp"},
            pop_size=1, selection_num=1, num_samplers=1, init_pop_size=1,
            reserve_sample_order=reserve, admit_sample_order=admit)
        self.assertEqual(calls, [0, 0, 0])
        self.assertEqual(admissions, [1])
        self.assertEqual(population[0]._recipe_sample_order, 1)

        # Initialization keeps the best candidates, rather than stopping at
        # the first population-sized group of feasible candidates.
        scores = iter((1.0, 3.0, 2.0))
        namespace["sample_initial_algorithm"] = lambda *a, **kw: (
            "algorithm", SimpleNamespace(initial_score=next(scores)))
        namespace["SecureEvaluator"] = lambda *a, **kw: SimpleNamespace(
            evaluate_program_record_time=lambda program: (program.initial_score, 0.1))
        population, _ = initialize(
            None, None, {"template_program": "template", "template_module": "tsp"},
            pop_size=1, selection_num=1, num_samplers=1, init_pop_size=3,
            reserve_sample_order=reserve)
        self.assertEqual([item.score for item in population], [3.0])


if __name__ == "__main__":
    unittest.main()
