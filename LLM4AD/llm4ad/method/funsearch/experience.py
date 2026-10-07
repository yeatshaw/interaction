from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
import uuid
from typing import Any

import numpy as np

from ...base import Function


class OpenAIEmbedding:
    """Small OpenAI-compatible text embedding client."""

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            encoding_format: str = 'float',
            timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.encoding_format = encoding_format
        self.timeout = int(timeout)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        request = urllib.request.Request(
            f'{self.base_url}/embeddings',
            data=json.dumps({
                'model': self.model,
                'input': inputs,
                'encoding_format': self.encoding_format,
            }).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise RuntimeError(f'Invalid embedding response: {str(payload)[:500]}')
        ordered = sorted(payload['data'], key=lambda item: item.get('index', 0))
        vectors = np.asarray([item['embedding'] for item in ordered], dtype=float)
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f'Embedding service returned {len(vectors)} vectors for {len(inputs)} inputs')
        return vectors[0] if single else vectors


class FunSearchExperienceStore:
    """Persist, retrieve, and distill RefineEvo-style experiences."""

    def __init__(
            self,
            reflector_llm,
            embedding_config: dict[str, Any] | None = None,
            top_k: int = 3,
            path: str | None = None,
    ) -> None:
        embedding_config = dict(embedding_config or {})
        self.reflector_llm = reflector_llm
        self.top_k = max(1, int(top_k))
        self.path = path
        self._lock = threading.RLock()
        self._experiences: list[dict[str, Any]] = []
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._embedding_error: str | None = None
        self.embedding_model = self._build_embedding(embedding_config)
        self._load()

    @property
    def embedding_available(self) -> bool:
        return self.embedding_model is not None

    @property
    def embedding_error(self) -> str | None:
        return self._embedding_error

    def build_query(
            self,
            reference: Function | None,
            parent_lineage: tuple[Function, ...],
            recipe_id: str,
    ) -> str:
        parts = [f'Recipe: {recipe_id}']
        if reference is not None:
            parts.append(f'Reference algorithm:\n{str(reference)}')
            parts.append(f'Reference score: {getattr(reference, "score", None)}')
        for index, parent in enumerate(parent_lineage, 1):
            parts.append(f'Parent algorithm {index}:\n{str(parent)}')
            parts.append(f'Parent score {index}: {getattr(parent, "score", None)}')
        return '\n\n'.join(parts)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        with self._lock:
            candidates = [
                item for item in self._experiences
                if item.get('experience_id') and item.get('query')
            ]
        if not candidates:
            return []
        if self.embedding_model is None:
            return candidates[-self.top_k:]
        try:
            query_vector = self._encode_one(query)
            missing = [
                item for item in candidates
                if item.get('experience_id') not in self._embedding_cache
            ]
            if missing:
                vectors = self._encode_many([item['query'] for item in missing])
                with self._lock:
                    self._embedding_cache.update({
                        item['experience_id']: vectors[index]
                        for index, item in enumerate(missing)
                    })
            with self._lock:
                matrix = np.asarray([
                    self._embedding_cache[item['experience_id']]
                    for item in candidates
                ], dtype=float)
            query_norm = np.linalg.norm(query_vector)
            matrix_norm = np.linalg.norm(matrix, axis=1)
            similarity = matrix.dot(query_vector) / np.maximum(
                matrix_norm * query_norm, 1e-12)
            indices = np.argsort(similarity)[-self.top_k:][::-1]
            return [candidates[int(index)] for index in indices]
        except Exception as exc:
            self._embedding_error = str(exc)
            return candidates[-self.top_k:]

    def distill(
            self,
            query: str,
            reference: Function | None,
            parent_lineage: tuple[Function, ...],
            child: Function,
            recipe_id: str,
    ) -> list[dict[str, Any]]:
        if self.reflector_llm is None or reference is None:
            return []
        parent_score = getattr(reference, 'score', None)
        child_score = getattr(child, 'score', None)
        trajectory = {
            'recipe_id': recipe_id,
            'reference': str(reference),
            'reference_score': parent_score,
            'parent_lineage': [str(parent) for parent in parent_lineage],
            'child': str(child),
            'child_score': child_score,
            'improvement': (
                float(child_score) - float(parent_score)
                if parent_score is not None and child_score is not None else None
            ),
        }
        try:
            response = self.reflector_llm.draw_sample(
                self._distillation_prompt(trajectory))
        except Exception:
            return []
        records = []
        for item in self._parse_items(response):
            records.append({
                'experience_id': str(uuid.uuid4()),
                'recipe_id': recipe_id,
                'summary': item['summary'],
                'recommendations': item['recommendations'],
                'applicable_when': item['applicable_when'],
                'query': query,
                'reference_score': parent_score,
                'child_score': child_score,
                'improvement': trajectory['improvement'],
                'trajectory': trajectory,
            })
        return records[:3]

    def add(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._lock:
            self._experiences.extend(records)
            if self.path:
                _atomic_json_dump(self.path, self._experiences)

    def prompt_text(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return '(no matching historical experience was retrieved)'
        sections = []
        for index, item in enumerate(records, 1):
            recommendations = '; '.join(item.get('recommendations', []))
            sections.append(
                f'Experience {index}:\n'
                f'Summary: {item.get("summary", "")}\n'
                f'Recommendations: {recommendations}\n'
                f'Applicable when: {item.get("applicable_when", "")}')
        return '\n\n'.join(sections)

    def _build_embedding(self, config: dict[str, Any]):
        if config.get('enabled', True) is False:
            return None
        api_key = config.get('api_key', '')
        if not api_key:
            env_names = config.get('api_key_env', [])
            if isinstance(env_names, str):
                env_names = [env_names]
            for env_name in env_names or []:
                api_key = os.environ.get(env_name, '')
                if api_key:
                    break
        if not api_key:
            self._embedding_error = 'embedding API key is empty'
            return None
        return OpenAIEmbedding(
            api_key=api_key,
            base_url=config.get('base_url', 'https://api.apilio.ai/v1'),
            model=config.get('model', 'text-embedding-v4'),
            encoding_format=config.get('encoding_format', 'float'),
            timeout=int(config.get('timeout', 60)),
        )

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embedding_model.encode(text), dtype=float)
        if vector.ndim != 1:
            raise ValueError(f'Expected one embedding vector, got shape {vector.shape}')
        return vector

    def _encode_many(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.embedding_model.encode(texts), dtype=float)
        if vectors.ndim == 2 and len(vectors) == len(texts):
            return vectors
        return np.asarray([self._encode_one(text) for text in texts], dtype=float)

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            if isinstance(data, list):
                self._experiences = [
                    item for item in data
                    if isinstance(item, dict)
                    and item.get('experience_id')
                    and item.get('query')
                ]
        except Exception:
            self._experiences = []

    @staticmethod
    def _distillation_prompt(trajectory: dict[str, Any]) -> str:
        return f'''You are distilling reusable RefineEvo-style experience from a FunSearch transition.

TRAJECTORY DATA:
{json.dumps(trajectory, ensure_ascii=False, indent=2)}

Output ONLY a valid JSON array containing 2-3 objects with exactly this schema:
[
  {{
    "summary": "one-sentence reusable lesson",
    "recommendations": ["action 1", "action 2"],
    "applicable_when": "clear condition for applying this lesson"
  }}
]
Do not output any text outside the JSON array.'''

    @staticmethod
    def _parse_items(response: str) -> list[dict[str, Any]]:
        if not isinstance(response, str):
            return []
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        valid = []
        for item in data if isinstance(data, list) else []:
            if (isinstance(item, dict)
                    and isinstance(item.get('summary'), str)
                    and isinstance(item.get('recommendations'), list)
                    and all(isinstance(value, str) for value in item['recommendations'])
                    and isinstance(item.get('applicable_when'), str)):
                valid.append({
                    'summary': item['summary'].strip(),
                    'recommendations': [
                        value.strip() for value in item['recommendations']
                    ],
                    'applicable_when': item['applicable_when'].strip(),
                })
        return valid[:3]


def _atomic_json_dump(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
