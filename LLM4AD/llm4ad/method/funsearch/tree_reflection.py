from __future__ import annotations

import copy
import dataclasses
import json
import os
import threading
from typing import Any

import numpy as np

from ...base import Function, TextFunctionProgramConverter
from .config import TreeReflectionConfig
from .experience import FunSearchExperienceStore


@dataclasses.dataclass
class FunSearchTreeNode:
    node_id: int
    island_id: int
    parent_node_id: int | None
    algorithm_id: int | None
    score: float | None
    depth: int
    parent_algorithm_ids: list[int] = dataclasses.field(default_factory=list)
    recipe_id: str | None = None
    experience_ids: list[str] = dataclasses.field(default_factory=list)
    children: list[int] = dataclasses.field(default_factory=list)
    is_root: bool = False
    function_path: str | None = None


@dataclasses.dataclass(frozen=True)
class ReflectionContext:
    island_id: int
    parent_node_id: int
    reference: Function | None
    reference_score: float | None
    state: tuple[float, float, float, float]
    recipe_id: str
    parent_lineage: tuple[Function, ...] = ()
    query: str = ''
    retrieved_experiences: tuple[dict[str, Any], ...] = ()


class IslandAlgorithmTree:
    """One independent algorithm tree for a FunSearch island."""

    def __init__(self, island_id: int) -> None:
        self.island_id = island_id
        self._next_node_id = 1
        self._pending_functions: dict[int, str] = {}
        self.nodes: dict[int, FunSearchTreeNode] = {
            0: FunSearchTreeNode(
                node_id=0,
                island_id=island_id,
                parent_node_id=None,
                algorithm_id=None,
                score=None,
                depth=0,
                is_root=True,
            )
        }

    def add_algorithm(
            self,
            function: Function,
            score: float | None,
            parent_node_id: int | None = None,
            recipe_id: str | None = None,
            experience_ids: list[str] | None = None,
    ) -> FunSearchTreeNode:
        if parent_node_id not in self.nodes:
            parent_node_id = 0
        parent = self.nodes[parent_node_id]
        node_id = self._next_node_id
        self._next_node_id += 1
        node = FunSearchTreeNode(
            node_id=node_id,
            island_id=self.island_id,
            parent_node_id=parent_node_id,
            algorithm_id=node_id,
            score=float(score) if score is not None else None,
            depth=parent.depth + 1,
            parent_algorithm_ids=(
                [parent.algorithm_id]
                if parent.algorithm_id is not None else []
            ),
            recipe_id=recipe_id,
            experience_ids=list(experience_ids or []),
        )
        self.nodes[node_id] = node
        self._pending_functions[node_id] = str(function)
        parent.children.append(node_id)
        return node

    def candidates(self) -> list[FunSearchTreeNode]:
        return [node for node in self.nodes.values() if not node.is_root]

    def select_reference(self, mode: str) -> FunSearchTreeNode | None:
        candidates = [node for node in self.candidates() if node.score is not None]
        if not candidates:
            return None
        if mode == 'random':
            return candidates[int(np.random.randint(len(candidates)))]
        if mode == 'lineage_parent':
            leaves = [node for node in candidates if not node.children]
            return max(leaves or candidates, key=lambda node: node.score)
        return max(candidates, key=lambda node: node.score)

    def get_lineage(self, node_id: int, max_items: int) -> list[FunSearchTreeNode]:
        lineage: list[FunSearchTreeNode] = []
        current = self.nodes.get(node_id)
        while current is not None and not current.is_root and len(lineage) < max_items:
            lineage.append(current)
            current = self.nodes.get(current.parent_node_id)
        return lineage

    def state(self) -> np.ndarray:
        scores = [node.score for node in self.candidates() if node.score is not None]
        if not scores:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        values = np.asarray(scores, dtype=float)
        return np.asarray([
            1.0,
            float(np.mean(values)),
            float(np.max(values)),
            float(np.var(values)),
        ], dtype=float)

    def pending_records(self) -> list[dict[str, Any]]:
        records = []
        for node_id in sorted(self._pending_functions):
            node = self.nodes[node_id]
            record = dataclasses.asdict(node)
            record['function'] = self._pending_functions[node_id]
            records.append(record)
        return records

    def mark_persisted(self, node_ids: list[int], function_path: str) -> None:
        for node_id in node_ids:
            node = self.nodes[node_id]
            node.function_path = function_path
            self._pending_functions.pop(node_id, None)

    def pending_function(self, node_id: int) -> str | None:
        return self._pending_functions.get(node_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], default_island_id: int) -> 'IslandAlgorithmTree':
        island_id = int(payload.get('island_id', default_island_id))
        tree = cls(island_id)
        tree._next_node_id = int(payload.get('next_node_id', 1))
        tree.nodes = {}
        for raw in payload.get('nodes', []):
            raw = dict(raw)
            legacy_function = raw.pop('function_text', None)
            node = FunSearchTreeNode(
                node_id=int(raw.get('node_id', 0)),
                island_id=int(raw.get('island_id', island_id)),
                parent_node_id=raw.get('parent_node_id'),
                algorithm_id=raw.get('algorithm_id'),
                score=raw.get('score'),
                depth=int(raw.get('depth', 0)),
                parent_algorithm_ids=list(raw.get('parent_algorithm_ids', [])),
                recipe_id=raw.get('recipe_id'),
                experience_ids=list(raw.get('experience_ids', [])),
                children=list(raw.get('children', [])),
                is_root=bool(raw.get('is_root', False)),
                function_path=raw.get('function_path'),
            )
            tree.nodes[node.node_id] = node
            if legacy_function:
                tree._pending_functions[node.node_id] = legacy_function
        if 0 not in tree.nodes:
            tree.nodes[0] = FunSearchTreeNode(
                node_id=0,
                island_id=island_id,
                parent_node_id=None,
                algorithm_id=None,
                score=None,
                depth=0,
                is_root=True,
            )
        return tree

    def to_dict(self) -> dict[str, Any]:
        return {
            'island_id': self.island_id,
            'next_node_id': self._next_node_id,
            'nodes': [dataclasses.asdict(node) for node in self.nodes.values()],
            'pending_node_ids': sorted(self._pending_functions),
        }


class ThompsonRecipeRouter:
    """Linear Gaussian Thompson sampler over reflection recipes."""

    def __init__(self, recipes: tuple[str, ...], config: TreeReflectionConfig) -> None:
        self._recipes = tuple(recipes)
        self._dimension = 4
        self._prior_precision = float(config.prior_precision)
        self._posterior_scale = float(config.posterior_scale)
        self._lock = threading.RLock()
        self._a = {
            recipe: self._prior_precision * np.eye(self._dimension, dtype=float)
            for recipe in self._recipes
        }
        self._b = {recipe: np.zeros(self._dimension, dtype=float) for recipe in self._recipes}
        self._trials = {recipe: 0 for recipe in self._recipes}

    def select(self, state: np.ndarray, adaptive: bool) -> str:
        with self._lock:
            if not adaptive:
                return self._recipes[0]
            sampled_rewards = {}
            for recipe in self._recipes:
                a_inv = np.linalg.pinv(self._a[recipe])
                mean = a_inv @ self._b[recipe]
                covariance = (self._posterior_scale ** 2) * a_inv
                theta = np.random.multivariate_normal(mean, covariance)
                sampled_rewards[recipe] = float(state @ theta)
            return max(sampled_rewards, key=sampled_rewards.get)

    def update(self, recipe_id: str, state: np.ndarray, reward: float) -> None:
        with self._lock:
            self._a[recipe_id] += np.outer(state, state)
            self._b[recipe_id] += float(reward) * state
            self._trials[recipe_id] += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                recipe: {
                    'A': self._a[recipe].tolist(),
                    'b': self._b[recipe].tolist(),
                    'trials': self._trials[recipe],
                }
                for recipe in self._recipes
            }

    def restore(self, payload: dict[str, Any]) -> None:
        with self._lock:
            for recipe in self._recipes:
                values = payload.get(recipe)
                if not values:
                    continue
                self._a[recipe] = np.asarray(values['A'], dtype=float)
                self._b[recipe] = np.asarray(values['b'], dtype=float)
                self._trials[recipe] = int(values.get('trials', 0))


class TreeReflectionManager:
    def __init__(
            self,
            num_islands: int,
            config: TreeReflectionConfig,
            reflector_llm=None,
            tree_path: str | None = None,
            stats_path: str | None = None,
            experience_path: str | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._trees = {island_id: IslandAlgorithmTree(island_id)
                       for island_id in range(num_islands)}
        self._router = ThompsonRecipeRouter(config.recipes, config)
        self._tree_path = tree_path
        self._stats_path = stats_path
        self._experience_path = experience_path
        self._tree_nodes_root = os.path.join(
            os.path.dirname(os.path.abspath(tree_path)), 'tree_nodes'
        ) if tree_path else None
        self._tree_persist_interval = max(1, int(config.tree_persist_interval))
        self._last_flush_sample = -1
        self._island_flush_orders = {island_id: 0 for island_id in self._trees}
        self._experience_store = None
        if 'refineevo_experience' in config.recipes:
            self._experience_store = FunSearchExperienceStore(
                reflector_llm=reflector_llm,
                embedding_config=config.embedding,
                top_k=config.experience_top_k,
                path=experience_path,
            )
            if not self._experience_store.embedding_available:
                print(
                    'WARNING: refineevo_experience embedding is unavailable; '
                    'historical experience retrieval will use fallback order. '
                    f'Reason: {self._experience_store.embedding_error}',
                    flush=True,
                )

    def initialize(self, function: Function, score: float) -> None:
        with self._lock:
            for tree in self._trees.values():
                tree.add_algorithm(copy.deepcopy(function), score, parent_node_id=0)
            self.flush(sample_order=0, force=True)

    def prepare(self, island_id: int) -> ReflectionContext:
        with self._lock:
            tree = self._trees[island_id]
            reference_node = tree.select_reference(self.config.reference_mode)
            if reference_node is None:
                reference = None
                reference_score = None
                parent_node_id = 0
            else:
                reference = self._node_to_function(tree, reference_node)
                reference_score = reference_node.score
                parent_node_id = reference_node.node_id
            lineage_nodes = tree.get_lineage(
                parent_node_id,
                max_items=self.config.max_parent_context + 1,
            )
            parent_lineage = tuple(
                self._node_to_function(tree, node) for node in lineage_nodes[1:]
            )
            state = tree.state()
            recipe_id = self._router.select(
                state, self.config.adaptive_recipe_selection)
            state_tuple = tuple(float(value) for value in state)

        # Embedding retrieval may perform network I/O and must stay outside
        # the tree lock so concurrent sampler threads remain independent.
        query = ''
        retrieved_experiences: tuple[dict[str, Any], ...] = ()
        if (recipe_id == 'refineevo_experience'
                and self._experience_store is not None):
            query = self._experience_store.build_query(
                reference, parent_lineage, recipe_id)
            retrieved_experiences = tuple(
                self._experience_store.retrieve(query))
        return ReflectionContext(
            island_id=island_id,
            parent_node_id=parent_node_id,
            reference=reference,
            reference_score=reference_score,
            state=state_tuple,
            recipe_id=recipe_id,
            parent_lineage=parent_lineage,
            query=query,
            retrieved_experiences=retrieved_experiences,
        )

    def commit(
            self,
            context: ReflectionContext,
            function: Function,
            score: float,
            sample_order: int | None = None,
    ) -> None:
        experience_records = []
        if (context.recipe_id == 'refineevo_experience'
                and self._experience_store is not None):
            experience_records = self._experience_store.distill(
                query=context.query,
                reference=context.reference,
                parent_lineage=context.parent_lineage,
                child=function,
                recipe_id=context.recipe_id,
            )
        experience_ids = [
            str(item['experience_id'])
            for item in experience_records
            if item.get('experience_id')
        ]
        if experience_records and self._experience_store is not None:
            for record in experience_records:
                record['island_id'] = context.island_id
                record['parent_node_id'] = context.parent_node_id
                record['reference_algorithm_id'] = context.parent_node_id
            self._experience_store.add(experience_records)
        with self._lock:
            reward = 0.0
            if context.reference_score is not None:
                reward = (float(score) - float(context.reference_score)) * self.config.reward_scale
            self._router.update(
                context.recipe_id,
                np.asarray(context.state, dtype=float),
                reward,
            )
            self._trees[context.island_id].add_algorithm(
                copy.deepcopy(function),
                score,
                parent_node_id=context.parent_node_id,
                recipe_id=context.recipe_id,
                experience_ids=experience_ids,
            )
            self.flush(sample_order=sample_order, force=False)

    def handle_database_reset(self, reset_records: list[tuple[int, Function | None, Any]]) -> None:
        with self._lock:
            for island_id, founder, founder_score in reset_records:
                self._trees[island_id] = IslandAlgorithmTree(island_id)
                if founder is not None and founder_score is not None:
                    self._trees[island_id].add_algorithm(
                        copy.deepcopy(founder), founder_score, parent_node_id=0)
            self.flush(sample_order=None, force=True)

    def _node_to_function(
            self,
            tree: IslandAlgorithmTree,
            node: FunSearchTreeNode,
    ) -> Function:
        function_text = tree.pending_function(node.node_id)
        if function_text is None and node.function_path:
            path = node.function_path
            if not os.path.isabs(path):
                path = os.path.join(
                    os.path.dirname(os.path.abspath(self._tree_path)), path)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    records = json.load(file)
                for record in records:
                    if int(record.get('node_id', -1)) == node.node_id:
                        function_text = record.get('function')
                        break
            except (OSError, json.JSONDecodeError):
                function_text = None
        function = TextFunctionProgramConverter.text_to_function(function_text or '')
        if function is None:
            raise ValueError(f'Invalid tree function for node {node.node_id}.')
        function.score = node.score
        return function

    def flush(
            self,
            sample_order: int | None = None,
            force: bool = False,
    ) -> tuple[str | None, str | None, str | None]:
        with self._lock:
            if (not force and sample_order is not None and
                    self._last_flush_sample >= 0 and
                    sample_order - self._last_flush_sample < self._tree_persist_interval):
                return self._tree_path, self._stats_path, self._experience_path
            if self._tree_nodes_root:
                for island_id, tree in self._trees.items():
                    records = tree.pending_records()
                    if not records:
                        continue
                    island_dir = os.path.join(
                        self._tree_nodes_root, f'island_{island_id:06d}')
                    os.makedirs(island_dir, exist_ok=True)
                    start = self._island_flush_orders[island_id] + 1
                    end = start + len(records) - 1
                    path = os.path.join(
                        island_dir, f'nodes_{start:06d}~{end:06d}.json')
                    _atomic_json_dump(path, records)
                    relative_path = os.path.relpath(
                        path, os.path.dirname(os.path.abspath(self._tree_path)))
                    tree.mark_persisted(
                        [int(record['node_id']) for record in records],
                        relative_path,
                    )
                    self._island_flush_orders[island_id] = end
            if sample_order is not None:
                self._last_flush_sample = int(sample_order)
            self._persist_locked()
            return self._tree_path, self._stats_path, self._experience_path

    def restore(self, tree_path: str | None = None, stats_path: str | None = None) -> bool:
        with self._lock:
            tree_path = tree_path or self._tree_path
            if not tree_path or not os.path.isfile(tree_path):
                return False
            try:
                with open(tree_path, 'r', encoding='utf-8') as file:
                    payload = json.load(file)
                tree_payload = payload.get('trees', payload)
                self._trees = {
                    int(island_id): IslandAlgorithmTree.from_dict(
                        tree_data, int(island_id))
                    for island_id, tree_data in tree_payload.items()
                }
                self._island_flush_orders = {
                    island_id: int(payload.get('island_flush_orders', {}).get(
                        str(island_id), 0))
                    for island_id in self._trees
                }
                self._last_flush_sample = int(payload.get('last_flush_sample', -1))
                stats_path = stats_path or self._stats_path
                if stats_path and os.path.isfile(stats_path):
                    with open(stats_path, 'r', encoding='utf-8') as file:
                        self._router.restore(json.load(file))
                return True
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return False

    def _persist_locked(self) -> None:
        if self._tree_path:
            payload = {
                'version': 2,
                'last_flush_sample': self._last_flush_sample,
                'island_flush_orders': {
                    str(island_id): order
                    for island_id, order in self._island_flush_orders.items()
                },
                'trees': {
                    str(island_id): tree.to_dict()
                    for island_id, tree in self._trees.items()
                },
            }
            _atomic_json_dump(self._tree_path, payload)
        if self._stats_path:
            _atomic_json_dump(self._stats_path, self._router.to_dict())


RECIPE_INSTRUCTIONS = {
    'full_difference_guidance': (
        'Perform a full difference analysis between the reference algorithm and its '
        'parent lineage. First compare their concrete design choices, then attribute '
        'the observed performance difference to those choices, and finally distill '
        'the most important implementation guidance for the next algorithm. Cover '
        'both beneficial and harmful differences.'),
    'attribution_both_conditions': (
        'Analyze both conditions of the reference algorithm: explain which design '
        'choices contribute to good performance and which choices create failure '
        'modes or regressions. State when each choice is likely to help or hurt, then '
        'provide conditional implementation guidance for the next algorithm.'),
    'attribution_good': (
        'Focus only on the successful design choices of the reference algorithm. '
        'Attribute its strong performance to concrete mechanisms, explain why they '
        'work for this task, and convert those mechanisms into actionable changes '
        'for the next implementation. Do not spend space on generic criticism.'),
    'comparison_only': (
        'Compare the reference algorithm directly with its parent lineage. Identify '
        'the most meaningful structural and behavioral differences and rank the '
        'changes that should be retained, removed, or refined. Do not speculate about '
        'unobserved causes; return comparison-based guidance only.'),
    'refineevo_experience': (
        'Distill reusable local experience from the reference algorithm and its parent '
        'lineage. Extract transferable lessons, likely failure modes, and a concrete '
        'rule for applying the experience when generating the next algorithm. Focus on '
        'experience that can be reused beyond this single reference algorithm.'),
    'comparison': (
        'Compare the reference algorithm with its parent lineage. Identify the most '
        'important strengths, weaknesses, and concrete changes that could improve the '
        'algorithm while preserving valid FunSearch interfaces.'),
    'attribution': (
        "Attribute the reference algorithm's observed performance to specific design "
        'choices. Separate beneficial choices from likely failure modes and propose '
        'actionable implementation changes.'),
    'summarization': (
        'Summarize the reusable optimization insight in the reference algorithm and '
        'turn it into concise guidance for generating a stronger implementation.'),
}


def build_reflection_prompt(
        context: ReflectionContext,
        manager: TreeReflectionManager,
    ) -> str:
    reference = str(context.reference) if context.reference is not None else '(empty root node)'
    recipe_instruction = RECIPE_INSTRUCTIONS.get(
        context.recipe_id,
        'Analyze the reference algorithm and provide concrete improvement guidance.',
    )
    lineage_text = '\n\n'.join(
        f'Parent algorithm {index}:\n{value}'
        for index, value in enumerate(
            (str(parent) for parent in context.parent_lineage), 1)
    ) or '(no additional parent context)'
    experience_text = '(experience retrieval is disabled)'
    if (context.recipe_id == 'refineevo_experience'
            and manager._experience_store is not None):
        experience_text = manager._experience_store.prompt_text(
            list(context.retrieved_experiences))
    return (
        'You are the reflection stage of a FunSearch algorithm search.\n'
        f'Reflection recipe: {context.recipe_id}\n'
        f'{recipe_instruction}\n\n'
        'Reference algorithm:\n'
        f'{reference}\n\n'
        'Additional parent lineage context:\n'
        f'{lineage_text}\n\n'
        'Retrieved RefineEvo-style historical experiences:\n'
        f'{experience_text}\n\n'
        'Use retrieved experiences only when they are relevant to the current '
        'reference algorithm and parent lineage.\n'
        'Return only concise, actionable natural-language guidance. Do not return a '
        'replacement function and do not include markdown fences.\n'
        'For refineevo_experience, explicitly include: reusable lesson, failure mode, '
        'transfer rule, and implementation guidance.\n'
    )


def augment_funsearch_prompt(base_prompt: str, reflection: str | None) -> str:
    if not reflection:
        return base_prompt
    return (
        '===== Tree Reflection Guidance =====\n'
        f'{reflection.strip()}\n'
        '===== End Tree Reflection Guidance =====\n\n'
        'Use the guidance above to improve the following FunSearch prompt.\n\n'
        f'{base_prompt}'
    )


def _atomic_json_dump(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
