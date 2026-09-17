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
            "token_usage": getattr(individual, "_recipe_token_usage", None),
            "sample_order": getattr(individual, "_recipe_sample_order", None),
            "code": (individual.to_code_without_docstring()
                     if hasattr(individual, "to_code_without_docstring")
                     else str(individual)),
            "recipe_id": getattr(individual, "_recipe_id", None),
        }

    def add(self, node_id, parent_id, depth, recipe_id, generation, population,
            experiences=None, token_usage=None):
        self.buffer.append({
            "population_node_id": node_id,
            "parent_population_node_id": parent_id,
            "depth": depth,
            "incoming_recipe_id": recipe_id,
            "generation": generation,
            "population_size": len(population.individuals),
            "best_score": max((x.score for x in population.individuals), default=float("-inf")),
            "token_usage": token_usage,
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

    @property
    def convergence_path(self):
        return os.path.join(self.directory, "overall_convergence.json")

    @property
    def convergence_plot_path(self):
        return os.path.join(self.directory, "overall_convergence.png")

    def load_convergence_records(self):
        try:
            with open(self.convergence_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        if isinstance(payload, dict):
            records = payload.get("records", [])
        elif isinstance(payload, list):
            records = payload
        else:
            records = []
        return records if isinstance(records, list) else []

    def write_convergence_records(self, records):
        payload = {
            "x_axis": "algorithm_id",
            "y_axis": "best_score_so_far",
            "records": list(records),
        }
        fd, tmp = tempfile.mkstemp(prefix=".overall_convergence_",
                                  dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.convergence_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def write_convergence_plot(self, records):
        points = []
        for record in records:
            if not isinstance(record, dict):
                continue
            algorithm_id = record.get("algorithm_id")
            best_score = record.get("best_score_so_far", record.get("best_score"))
            if algorithm_id is None or best_score is None:
                continue
            try:
                points.append((int(algorithm_id), float(best_score)))
            except (TypeError, ValueError):
                continue
        if not points:
            return None
        points.sort(key=lambda item: item[0])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_values = [item[0] for item in points]
        y_values = [item[1] for item in points]
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.step(x_values, y_values, where="post", linewidth=1.8)
        ax.scatter([x_values[-1]], [y_values[-1]], s=26, zorder=3)
        ax.set_xlabel("Algorithm ID")
        ax.set_ylabel("Best score so far")
        ax.set_title("Global convergence")
        ax.grid(True, alpha=0.28)
        if len(x_values) == 1:
            ax.set_xlim(x_values[0] - 1, x_values[0] + 1)
        fig.tight_layout()
        fig.savefig(self.convergence_plot_path, dpi=200)
        plt.close(fig)
        return self.convergence_plot_path

    @property
    def elite_pool_path(self):
        return os.path.join(self.directory, "global_elite_pool.json")

    def load_elite_pool_records(self):
        try:
            with open(self.elite_pool_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        if isinstance(payload, dict):
            records = payload.get("algorithms", [])
        elif isinstance(payload, list):
            records = payload
        else:
            records = []
        return records if isinstance(records, list) else []

    def write_elite_pool(self, individuals, elite_pool_size):
        """Atomically replace the optional global elite algorithm pool."""
        payload = {
            "elite_pool_size": int(elite_pool_size),
            "population_size": len(individuals),
            "algorithms": [self._algorithm_record(x) for x in individuals],
        }
        fd, tmp = tempfile.mkstemp(prefix=".global_elite_pool_", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.elite_pool_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
