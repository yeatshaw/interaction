from __future__ import annotations

import json
import os
import tempfile


class RecipeStore:
    """Batch persistence for population snapshots and MCTS checkpoints."""

    def __init__(self, directory, batch_size=10):
        self.directory = os.path.abspath(directory)
        self.batch_size = int(batch_size)
        self.buffer = []
        self.next_file_start = 1
        os.makedirs(self.directory, exist_ok=True)
        for name in os.listdir(self.directory):
            if name.startswith("population_nodes_") and name.endswith(".json"):
                try:
                    self.next_file_start = max(
                        self.next_file_start,
                        int(name[:-5].rsplit("~", 1)[1]) + 1)
                except (ValueError, IndexError):
                    pass

    @staticmethod
    def _algorithm_record(individual):
        return {
            "algorithm_id": getattr(individual, "_recipe_algorithm_id", None),
            "parent_algorithm_ids": list(getattr(individual, "_eoh_parent_ids", ())),
            "score": getattr(individual, "score", None),
            "thought": getattr(individual, "algorithm", ""),
            "suggestion": getattr(individual, "_eoh_generation_suggestion", None),
            "experience": getattr(individual, "_eoh_experience", None),
            "operator": getattr(individual, "operator", None),
            "evaluate_time": getattr(individual, "evaluate_time", None),
            "sample_time": getattr(individual, "sample_time", None),
            "code": (individual.to_code_without_docstring()
                     if hasattr(individual, "to_code_without_docstring")
                     else str(individual)),
            "recipe_id": getattr(individual, "_recipe_id", None),
        }

    def add(self, node_id, parent_id, depth, recipe_id, generation, population,
            experiences=None):
        self.buffer.append({
            "population_node_id": node_id,
            "parent_population_node_id": parent_id,
            "depth": depth,
            "incoming_recipe_id": recipe_id,
            "generation": generation,
            "population_size": len(population.individuals),
            "best_score": max((x.score for x in population.individuals), default=float("-inf")),
            "algorithms": [self._algorithm_record(x) for x in population.individuals],
            "experiences": list(experiences or []),
        })
        if len(self.buffer) >= self.batch_size:
            return self.flush()
        return None

    def flush(self):
        if not self.buffer:
            return None
        start = self.next_file_start
        end = start + len(self.buffer) - 1
        path = os.path.join(self.directory, f"population_nodes_{start:06d}~{end:06d}.json")
        fd, tmp = tempfile.mkstemp(prefix=".population_nodes_", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"nodes": self.buffer}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.next_file_start = end + 1
        self.buffer = []
        return path

    def load_node(self, node_id):
        """Load one node without retaining all historical populations in RAM."""
        node_id = int(node_id)
        for record in self.buffer:
            if int(record["population_node_id"]) == node_id:
                return record
        prefix = "population_nodes_"
        for name in os.listdir(self.directory):
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            bounds = name[len(prefix):-5].split("~")
            if len(bounds) != 2 or not (int(bounds[0]) <= node_id <= int(bounds[1])):
                continue
            with open(os.path.join(self.directory, name), encoding="utf-8") as f:
                for record in json.load(f).get("nodes", []):
                    if int(record["population_node_id"]) == node_id:
                        return record
        raise KeyError(f"Population node {node_id} is not persisted")

    def load_algorithm_records(self, algorithm_ids):
        """Resolve selected lineage parents without retaining historical nodes."""
        wanted = {int(value) for value in algorithm_ids if value is not None}
        found = {}

        def inspect(nodes):
            for node in nodes:
                for item in node.get("algorithms", []):
                    algorithm_id = item.get("algorithm_id")
                    if algorithm_id is not None and int(algorithm_id) in wanted:
                        found[int(algorithm_id)] = item

        inspect(self.buffer)
        for name in os.listdir(self.directory):
            if len(found) == len(wanted):
                break
            if not name.startswith("population_nodes_") or not name.endswith(".json"):
                continue
            with open(os.path.join(self.directory, name), encoding="utf-8") as f:
                inspect(json.load(f).get("nodes", []))
        return found

    def write_checkpoint(self, path, state):
        fd, tmp = tempfile.mkstemp(prefix=".checkpoint_", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @property
    def best_sample_path(self):
        return os.path.join(self.directory, "sample_best.json")

    def load_best_samples(self):
        try:
            with open(self.best_sample_path, encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def write_best_samples(self, records):
        """Atomically replace the small global-best history file."""
        fd, tmp = tempfile.mkstemp(prefix=".sample_best_", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.best_sample_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
