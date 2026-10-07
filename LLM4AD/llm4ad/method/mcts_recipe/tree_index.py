from __future__ import annotations

import json
import os
import tempfile


class DiskTreeIndex:
    """Atomic JSON index shared by disk-backed MCTS implementations."""

    def __init__(self, directory, filename="tree_state.json"):
        self.directory = os.path.abspath(directory)
        self.path = os.path.join(self.directory, filename)

    def load(self, node_id_field):
        try:
            with open(self.path, encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return {"root_id": None, "nodes": []}
        if not isinstance(payload, dict):
            return {"root_id": None, "nodes": []}
        nodes = payload.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []
        return {
            "root_id": payload.get("root_id"),
            "nodes": [self._normalize(item, node_id_field)
                      for item in nodes if isinstance(item, dict)
                      and (item.get(node_id_field) is not None
                           or item.get("node_id") is not None)],
        }

    @staticmethod
    def _normalize(record, node_id_field="node_id"):
        item = dict(record)
        if node_id_field == "population_node_id":
            node_id = item.get("population_node_id", item.get("node_id"))
            parent_id = item.get(
                "parent_population_node_id", item.get("parent_node_id"))
            children = item.get(
                "children_population_node_ids", item.get("children_node_ids", []))
        else:
            node_id = item.get("node_id", item.get("population_node_id"))
            parent_id = item.get(
                "parent_node_id", item.get("parent_population_node_id"))
            children = item.get(
                "children_node_ids", item.get("children_population_node_ids", []))
        item["node_id"] = node_id
        item["population_node_id"] = node_id
        item["parent_node_id"] = parent_id
        item["parent_population_node_id"] = parent_id
        item["children_node_ids"] = list(children or [])
        item["children_population_node_ids"] = list(children or [])
        return item

    def write(self, root_id, nodes, node_id_field="node_id"):
        os.makedirs(self.directory, exist_ok=True)
        payload = {
            "root_id": int(root_id) if root_id is not None else None,
            "nodes": [self._normalize(item, node_id_field) for item in nodes],
        }
        fd, temp_path = tempfile.mkstemp(
            prefix=".tree_state_", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
