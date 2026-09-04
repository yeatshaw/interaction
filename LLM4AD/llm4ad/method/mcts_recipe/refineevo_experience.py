"""Node-local RefineEvo-style experience distillation and retrieval."""

from __future__ import annotations

import json
import re
import threading
import uuid

import numpy as np


OPERATOR_GOALS = {
    "e1": "Please refer to the given suggestions and help me create a new algorithm that has a totally different form from the given ones.",
    "e2": "Please refer to the given suggestions and identify the common backbone idea in the provided methods and help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.",
    "m1": "Please refer to the given suggestions and assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.",
    "m2": "Please refer to the given suggestions and identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the score function provided.",
}


class RefineEvoExperienceManager:
    """Match current parent context to historical trajectory experiences.

    This follows RefineEvo's original retrieval semantics: embeddings compare
    parent query contexts. Vectors are computed once when a node is expanded
    and are not persisted in every descendant node.
    """

    def __init__(self, reflector_llm, embedding_model, top_k=3):
        self.reflector_llm = reflector_llm
        self.embedding_model = embedding_model
        self.top_k = int(top_k)
        self._node_embedding_cache = {}
        self._cache_lock = threading.RLock()

    def begin_node(self, experiences):
        """Build a temporary cache for one node expansion only."""
        items = [x for x in experiences if x.get("experience_id") and x.get("query")]
        with self._cache_lock:
            self._node_embedding_cache = {}
        if not items:
            return
        vectors = self._encode_many([x["query"] for x in items])
        with self._cache_lock:
            self._node_embedding_cache = {
                item["experience_id"]: vectors[index]
                for index, item in enumerate(items)
            }

    def end_node(self):
        with self._cache_lock:
            self._node_embedding_cache = {}

    @staticmethod
    def build_query(refs, operator):
        lines = [f"Operator goal: {OPERATOR_GOALS[operator]}"]
        for i, ref in enumerate(refs, 1):
            thought = re.sub(r"\s+", " ", str(getattr(ref, "algorithm", ""))).strip()
            lines.append(f"Parent {i} (score={getattr(ref, 'score', None)}): {thought[:400]}")
        return "\n".join(lines)

    def retrieve(self, refs, operator, experiences):
        candidates = [x for x in experiences
                      if x.get("operator_goal") == OPERATOR_GOALS[operator]
                      and x.get("query")]
        if not candidates:
            return []
        query_vector = np.asarray(
            self.embedding_model.encode(self.build_query(refs, operator)), dtype=float)
        with self._cache_lock:
            missing = [x for x in candidates
                       if x["experience_id"] not in self._node_embedding_cache]
        if missing:
            vectors = self._encode_many([x["query"] for x in missing])
            with self._cache_lock:
                self._node_embedding_cache.update({
                    item["experience_id"]: vectors[index]
                    for index, item in enumerate(missing)
                })
        with self._cache_lock:
            matrix = np.asarray([
                self._node_embedding_cache[x["experience_id"]] for x in candidates
            ], dtype=float)
        q_norm = np.linalg.norm(query_vector)
        norms = np.linalg.norm(matrix, axis=1)
        similarity = matrix.dot(query_vector) / np.maximum(norms * q_norm, 1e-12)
        indices = np.argsort(similarity)[-self.top_k:][::-1]
        return [candidates[int(i)] for i in indices]

    def _encode_many(self, texts):
        """Embed current and historical queries without persisting vectors."""
        try:
            vectors = np.asarray(self.embedding_model.encode(texts), dtype=float)
            if vectors.ndim == 2 and len(vectors) == len(texts):
                return vectors
        except Exception:
            pass
        return np.asarray([self.embedding_model.encode(text) for text in texts],
                          dtype=float)

    def distill(self, refs, operator):
        """Distill completed lineage transitions before generating a new child."""
        query = self.build_query(refs, operator)
        records = []
        for child in refs:
            parents = list(getattr(child, "_recipe_parent_functions", ()))
            parent_scores = [float(x.score) for x in parents if x.score is not None]
            is_success = (float(child.score) > max(parent_scores)
                          if parent_scores and child.score is not None else None)

            def algorithm_data(function):
                if hasattr(function, "to_code_without_docstring"):
                    code = function.to_code_without_docstring().rstrip()
                else:
                    code = str(function).rstrip()
                return {
                    "thought": str(getattr(function, "algorithm", "") or "").strip(),
                    "code": code,
                    "score": function.score,
                }

            trajectory = {
                "operator_goal": OPERATOR_GOALS[operator],
                "parents": [algorithm_data(parent) for parent in parents],
                "child": algorithm_data(child),
                "objective_type": "max",
                "best_parent_score": max(parent_scores) if parent_scores else None,
                "improvement": (float(child.score) - max(parent_scores)
                                if parent_scores and child.score is not None else None),
            }
            prompt = self._distillation_prompt(trajectory, is_success)
            response = self.reflector_llm.draw_sample(prompt)
            items = self._parse_items(response)
            for item in items:
                records.append({
                    "experience_id": str(uuid.uuid4()),
                    "is_success": is_success,
                    "summary": item["summary"],
                    "recommendations": item["recommendations"],
                    "applicable_when": item["applicable_when"],
                    "operator_goal": OPERATOR_GOALS[operator],
                    "query": query,
                    "score": 0,
                    "trajectory": trajectory,
                })
        return records

    @staticmethod
    def _distillation_prompt(trajectory, is_success):
        if is_success is None:
            opening = ("The reference algorithm has no recorded lineage parent. "
                       "Analyze its design and distill potentially useful experience "
                       "without claiming that it improved over a parent.")
        elif is_success:
            opening = ("The following parent-to-child algorithm evolution attempt "
                       "successfully improved performance. Analyze what worked well.")
        else:
            opening = ("The following parent-to-child algorithm evolution attempt "
                       "failed to improve performance. Analyze what went wrong.")
        return f'''You are an expert algorithmic researcher. {opening}

TRAJECTORY DATA:
```json
{json.dumps(trajectory, ensure_ascii=False, indent=2)}
```

Output ONLY a valid JSON array containing 2-3 objects with exactly this schema:
[
  {{
    "summary": "one-sentence lesson",
    "recommendations": ["action 1", "action 2"],
    "applicable_when": "clear condition for using this lesson"
  }}
]
Do not output markdown fences or any text outside the JSON array.'''

    @staticmethod
    def _parse_items(response):
        if not isinstance(response, str):
            return []
        match = re.search(r"\[[\s\S]*\]", response)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        valid = []
        for item in data if isinstance(data, list) else []:
            if (isinstance(item, dict)
                    and isinstance(item.get("summary"), str)
                    and isinstance(item.get("recommendations"), list)
                    and all(isinstance(x, str) for x in item["recommendations"])
                    and isinstance(item.get("applicable_when"), str)):
                valid.append({
                    "summary": item["summary"].strip(),
                    "recommendations": [x.strip() for x in item["recommendations"]],
                    "applicable_when": item["applicable_when"].strip(),
                })
        return valid[:3]
