from __future__ import annotations

import copy
import threading

from ..mcts_recipe.eoh_adapter import RefineEvoRecipeExpander


class AdaptiveRefineEvoReflector:
    """Reuse the existing RefineEvo prompts and retrieval on one EoH path."""

    def __init__(self, experience_manager, lineage_lookup):
        self.manager = experience_manager
        self.lineage_lookup = lineage_lookup
        self._experiences = []
        self._lock = threading.RLock()

    def __call__(self, refs, population, config, operator):
        refs = copy.deepcopy(list(refs))
        for ref in refs:
            ref._recipe_parent_functions = tuple(self.lineage_lookup(ref))
        with self._lock:
            snapshot = copy.deepcopy(self._experiences)
        retrieved = self.manager.retrieve(refs, operator, snapshot)
        fresh = self.manager.distill(refs, operator)
        fresh = fresh if isinstance(fresh, list) else ([fresh] if fresh else [])
        guidance = RefineEvoRecipeExpander._format_experiences(
            list(retrieved) + fresh)
        state = {
            "retrieved_ids": [item.get("experience_id") for item in retrieved
                              if isinstance(item, dict)],
            "fresh": fresh,
        }
        return guidance or None, state

    def on_result(self, state, improved):
        if not state:
            return
        retrieved_ids = set(state.get("retrieved_ids", ()))
        with self._lock:
            retained = []
            for item in self._experiences:
                if not isinstance(item, dict):
                    retained.append(item)
                    continue
                if item.get("experience_id") in retrieved_ids:
                    item["score"] = int(item.get("score", 0)) + (
                        1 if improved else -1)
                if int(item.get("score", 0)) >= 0:
                    retained.append(item)
            retained.extend(copy.deepcopy(state.get("fresh", ())))
            self._experiences = retained

