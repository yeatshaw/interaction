from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import threading

from ...base import TextFunctionProgramConverter
from ..mcts_recipe.tree_index import DiskTreeIndex


class MCTSAHDRecipeStore:
    """Persist MCTS-AHD recipe nodes in batches and write resumable state."""

    def __init__(self, directory, batch_size=100):
        self.directory = os.path.abspath(directory)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.nodes_directory = os.path.join(self.directory, "tree_nodes")
        os.makedirs(self.nodes_directory, exist_ok=True)
        self._tree_index = DiskTreeIndex(self.directory)
        self._lock = threading.RLock()
        self.buffer = []
        self.next_file_start = 1
        pattern = re.compile(r"^nodes_(\d+)~(\d+)\.json$")
        for name in os.listdir(self.nodes_directory):
            match = pattern.match(name)
            if match:
                self.next_file_start = max(
                    self.next_file_start, int(match.group(2)) + 1)

    @staticmethod
    def _algorithm_record(function):
        code = (function.to_code_without_docstring()
                if hasattr(function, "to_code_without_docstring")
                else str(function))
        return {
            "algorithm_id": getattr(function, "_recipe_algorithm_id", None),
            "node_id": getattr(function, "_mcts_ahd_node_id", None),
            "parent_algorithm_ids": list(
                getattr(function, "_eoh_parent_ids", ())),
            "score": getattr(function, "score", None),
            "thought": getattr(function, "algorithm", ""),
            "operator": getattr(function, "operator", None),
            "recipe_id": getattr(function, "_recipe_id", None),
            "sample_order": getattr(function, "_recipe_sample_order", None),
            "token_usage": getattr(function, "_recipe_token_usage", None),
            "experiences": list(getattr(
                function, "_refineevo_experiences", []) or []),
            "code": code,
        }

    def node_record(self, node, refs=None, experiences=None):
        function = getattr(node, "individual", None)
        refs = list(refs or [])
        return {
            "node_id": int(getattr(node, "node_id")),
            "parent_node_id": (
                int(getattr(node.parent, "node_id"))
                if getattr(node, "parent", None) is not None else None
            ),
            "depth": int(getattr(node, "depth", 0)),
            "Q": getattr(node, "Q", None),
            "visits": int(getattr(node, "visits", 0)),
            "children_node_ids": [
                int(getattr(child, "node_id"))
                for child in getattr(node, "children", [])
                if getattr(child, "node_id", None) is not None
            ],
            "algorithm": self._algorithm_record(function) if function else None,
            "parent_algorithm_ids": [
                getattr(ref, "_recipe_algorithm_id", None) for ref in refs
            ],
            "parent_node_ids": [
                getattr(ref, "_mcts_ahd_node_id", None) for ref in refs
            ],
            "recipe_id": getattr(function, "_recipe_id", None) if function else None,
            "operator": getattr(function, "operator", None) if function else None,
            "sample_order": (
                getattr(function, "_recipe_sample_order", None)
                if function else None
            ),
            "token_usage": (
                getattr(function, "_recipe_token_usage", None)
                if function else getattr(node, "token_usage", None)
            ),
            "experiences": list(experiences or getattr(
                node, "_refineevo_experiences", []) or []),
        }

    def add(self, record):
        with self._lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.batch_size:
                return self._flush_locked()
        return None

    def flush(self):
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self):
        if not self.buffer:
            return None
        start = self.next_file_start
        end = start + len(self.buffer) - 1
        path = os.path.join(
            self.nodes_directory,
            f"nodes_{start:06d}~{end:06d}.json",
        )
        fd, temp_path = tempfile.mkstemp(
            prefix=".nodes_", suffix=".json", dir=self.nodes_directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"nodes": self.buffer}, stream,
                          ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        self.next_file_start = end + 1
        self.buffer = []
        return path

    def load_all_nodes(self):
        records = {}
        with self._lock:
            paths = sorted(
                os.path.join(self.nodes_directory, name)
                for name in os.listdir(self.nodes_directory)
                if name.startswith("nodes_") and name.endswith(".json")
            )
            for path in paths:
                try:
                    with open(path, encoding="utf-8") as stream:
                        payload = json.load(stream)
                except (OSError, json.JSONDecodeError):
                    continue
                for record in payload.get("nodes", []):
                    if record.get("node_id") is not None:
                        records[int(record["node_id"])] = record
            for record in self.buffer:
                if record.get("node_id") is not None:
                    records[int(record["node_id"])] = record
        return records

    def load_node(self, node_id):
        node_id = int(node_id)
        with self._lock:
            for record in self.buffer:
                if int(record.get("node_id", -1)) == node_id:
                    return record
            for name in sorted(os.listdir(self.nodes_directory)):
                if not name.startswith("nodes_") or not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(self.nodes_directory, name),
                              encoding="utf-8") as stream:
                        records = json.load(stream).get("nodes", [])
                except (OSError, json.JSONDecodeError):
                    continue
                for record in records:
                    if int(record.get("node_id", -1)) == node_id:
                        return record
        raise KeyError(f"MCTS-AHD node {node_id} is not persisted")

    def load_algorithm_records(self, algorithm_ids):
        wanted = {int(value) for value in algorithm_ids if value is not None}
        found = {}

        def inspect(nodes):
            for node in nodes:
                algorithm = node.get("algorithm") or {}
                algorithm_id = algorithm.get("algorithm_id")
                if algorithm_id is not None and int(algorithm_id) in wanted:
                    found[int(algorithm_id)] = algorithm

        with self._lock:
            inspect(self.buffer)
            for name in os.listdir(self.nodes_directory):
                if len(found) == len(wanted):
                    break
                if not name.startswith("nodes_") or not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(self.nodes_directory, name),
                              encoding="utf-8") as stream:
                        inspect(json.load(stream).get("nodes", []))
                except (OSError, json.JSONDecodeError):
                    continue
        return found

    @property
    def tree_state_path(self):
        return os.path.join(self.directory, "tree_state.json")

    def load_tree_state(self):
        """Load only the disk-backed MCTS metadata index."""
        return self._tree_index.load("node_id")

    def write_tree_state(self, root_id, nodes):
        self._tree_index.write(root_id, nodes, node_id_field="node_id")

    @staticmethod
    def function_from_record(record):
        function = TextFunctionProgramConverter.text_to_function(record["code"])
        if function is None:
            raise ValueError(
                f"Invalid persisted algorithm {record.get('algorithm_id')}")
        function.score = record.get("score")
        function.algorithm = record.get("thought", "")
        function.operator = record.get("operator")
        function._recipe_algorithm_id = record.get("algorithm_id")
        function._mcts_ahd_node_id = record.get("node_id")
        function._eoh_parent_ids = tuple(record.get("parent_algorithm_ids", ()))
        function._recipe_id = record.get("recipe_id")
        function._recipe_sample_order = record.get("sample_order")
        function._recipe_token_usage = record.get("token_usage")
        function._refineevo_experiences = list(
            record.get("experiences", []) or [])
        return function

    def write_checkpoint(self, index, state):
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"checkpoint_{index:06d}.json")
        latest = os.path.join(self.directory, "checkpoint_latest.json")
        for target in (path, latest):
            fd, temp_path = tempfile.mkstemp(
                prefix=".checkpoint_", suffix=".json", dir=self.directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(state, stream, ensure_ascii=False, indent=2)
                os.replace(temp_path, target)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        return path

    def latest_checkpoint(self):
        path = os.path.join(self.directory, "checkpoint_latest.json")
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream), path
        except (OSError, json.JSONDecodeError):
            return None, None

    def write_token_usage(self, usage):
        path = os.path.join(self.directory, "token_usage_total.json")
        fd, temp_path = tempfile.mkstemp(
            prefix=".token_usage_", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(usage, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return path

    def write_convergence(self, records):
        json_path = os.path.join(self.directory, "convergence.json")
        csv_path = os.path.join(self.directory, "convergence.csv")
        payload = {
            "x_axis": "sample_order",
            "y_axis": "best_so_far",
            "records": list(records),
        }
        fd, temp_path = tempfile.mkstemp(
            prefix=".convergence_", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, json_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        fd, temp_path = tempfile.mkstemp(
            prefix=".convergence_", suffix=".csv", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["sample_order", "best_so_far", "score"])
                writer.writeheader()
                for record in records:
                    writer.writerow({
                        "sample_order": record.get("sample_order"),
                        "best_so_far": record.get("best_so_far"),
                        "score": record.get("score"),
                    })
            os.replace(temp_path, csv_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return json_path, csv_path

    def write_convergence_plot(self, records):
        points = []
        for record in records:
            try:
                points.append((int(record["sample_order"]),
                               float(record["best_so_far"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not points:
            return None
        points.sort(key=lambda item: item[0])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_values = [item[0] for item in points]
        y_values = [item[1] for item in points]
        fig, axis = plt.subplots(figsize=(9.5, 5.5))
        axis.step(x_values, y_values, where="post", linewidth=1.8)
        axis.set_xlabel("sample_order")
        axis.set_ylabel("best_so_far")
        axis.set_title("MCTS-AHD-Recipe convergence")
        axis.grid(True, alpha=0.28)
        fig.tight_layout()
        path = os.path.join(self.directory, "convergence.png")
        fd, temp_path = tempfile.mkstemp(
            prefix=".convergence_", suffix=".png", dir=self.directory)
        os.close(fd)
        try:
            fig.savefig(temp_path, dpi=200)
            os.replace(temp_path, path)
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return path
