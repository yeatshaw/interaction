from __future__ import annotations

import math


class RecipeNode:
    """Lightweight in-memory tree node; full populations live on disk."""

    def __init__(self, population_node_id, q, depth=0, parent=None,
                 incoming_recipe_id=None, visits=1):
        self.population_node_id = int(population_node_id)
        self.parent_population_node_id = (
            parent.population_node_id if parent is not None else None)
        self.depth = int(depth)
        self.Q = float(q)
        self.visits = int(visits)
        self.parent = parent
        self.children = []
        self.expanded_recipe_ids = set()
        self.incoming_recipe_id = incoming_recipe_id

    def add_child(self, child):
        self.children.append(child)
        self.expanded_recipe_ids.add(child.incoming_recipe_id)


class RecipeMCTS:
    def __init__(self, recipes, exploration_constant=0.1, max_depth=50):
        self.recipes = dict(recipes)
        self.exploration_constant = float(exploration_constant)
        self.max_depth = int(max_depth)
        self.root = None

    @staticmethod
    def _normalize(q, q_min, q_max):
        if q_max <= q_min:
            return 0.0
        return (q - q_min) / (q_max - q_min)

    def uct(self, node, q_min=None, q_max=None):
        if node.parent is None:
            return float("inf")
        siblings = node.parent.children
        q_min = min((x.Q for x in siblings), default=node.Q) if q_min is None else q_min
        q_max = max((x.Q for x in siblings), default=node.Q) if q_max is None else q_max
        exploit = self._normalize(node.Q, q_min, q_max)
        explore = self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1.0) / max(node.visits, 1))
        return exploit + explore

    def select(self):
        """Select a leaf by UCT; every selected leaf is expanded only once."""
        node = self.root
        while node.children and node.depth < self.max_depth:
            if len(node.expanded_recipe_ids) < len(self.recipes):
                return node
            node = max(node.children, key=self.uct)
        return node

    def backpropagate(self, node):
        child = node
        while child.parent is not None:
            parent = child.parent
            parent.Q = max(parent.Q, child.Q)
            parent.visits += 1
            child = parent
