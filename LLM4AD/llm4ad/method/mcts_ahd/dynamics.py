from __future__ import annotations

import json
import math
import os
from collections import defaultdict


class SearchDynamicsRecorder:
    """Records expansion events and round-level MCTS search states."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.round_id = 0
        self._round_events = []
        self._last_best_score = None
        self._last_tree_size = 0
        self._stagnation_length = 0
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.event_path = os.path.join(log_dir, 'expansion_events.jsonl')
            self.state_path = os.path.join(log_dir, 'search_states.jsonl')

    @property
    def enabled(self):
        return self.log_dir is not None

    def start_round(self):
        self.round_id += 1

    def discard_pending_events(self):
        """Exclude initialization events from round-level search metrics."""
        self._round_events = []

    def restore_from_logs(self):
        """Restore round numbering and counters when appending to existing logs."""
        if not self.enabled:
            return 0, 0
        events = self._load_jsonl(self.event_path)
        states = self._load_jsonl(self.state_path)
        if states:
            self.round_id = max(state.get('round_id', 0) for state in states)
            self._last_best_score = states[-1].get('best_score')
            self._last_tree_size = states[-1].get('tree_size', 0)
            self._stagnation_length = states[-1].get('stagnation_length', 0)
        attempts = [event for event in events if event.get('event_type') == 'attempt']
        evaluations = [event for event in attempts if event.get('status') in {'evaluated', 'evaluation_failed'}]
        return len(attempts), len(evaluations)

    def record_event(self, event):
        event = {'round_id': self.round_id, **event}
        self._round_events.append(event)
        if self.enabled:
            self._append_jsonl(self.event_path, event)

    def record_state(self, mcts, sample_count, evaluation_count):
        nodes = self._walk_tree(mcts.root)
        non_root = [node for node in nodes if node is not mcts.root]
        scores = [node.Q for node in non_root]
        best_score = max(scores) if scores else None
        if self._last_best_score is None or best_score is None:
            best_improvement = 0.0
            self._stagnation_length = 0
        else:
            best_improvement = best_score - self._last_best_score
            self._stagnation_length = 0 if best_improvement > 0 else self._stagnation_length + 1

        attempts = [event for event in self._round_events if event['event_type'] == 'attempt']
        outcomes = [event for event in self._round_events if event['event_type'] == 'outcome']
        accepted = [event for event in outcomes if event.get('accepted')]
        deltas = [event['score_delta'] for event in accepted if event.get('score_delta') is not None]
        root_entropy, branch_concentration = self._visit_statistics(
            [child.visits for child in mcts.root.children]
        )

        state = {
            'round_id': self.round_id,
            'samples': sample_count,
            'evaluations': evaluation_count,
            'best_score': best_score,
            'best_improvement': best_improvement,
            'mean_parent_child_gain': sum(deltas) / len(deltas) if deltas else None,
            'improvement_rate': sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0,
            'valid_rate': len(accepted) / len(attempts) if attempts else 0.0,
            'duplicate_rate': (
                sum(event.get('duplicate', False) for event in outcomes) / len(outcomes)
                if outcomes else 0.0
            ),
            'root_visit_entropy': root_entropy,
            'branch_concentration': branch_concentration,
            'tree_size': len(non_root),
            'tree_growth': len(non_root) - self._last_tree_size,
            'max_depth': max((node.depth for node in non_root), default=0),
            'stagnation_length': self._stagnation_length,
            'operator_stats': self._operator_statistics(attempts, accepted),
        }
        if self.enabled:
            self._append_jsonl(self.state_path, state)
        self._last_best_score = best_score
        self._last_tree_size = len(non_root)
        self._round_events = []
        return state

    @staticmethod
    def _walk_tree(root):
        nodes = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    @staticmethod
    def _visit_statistics(visits):
        total = sum(visits)
        if not visits:
            return 0.0, 0.0
        if len(visits) == 1 or total == 0:
            return 0.0, 1.0
        probabilities = [visit / total for visit in visits if visit > 0]
        entropy = -sum(p * math.log(p) for p in probabilities) / math.log(len(visits))
        return entropy, max(probabilities)

    @staticmethod
    def _operator_statistics(attempts, accepted):
        attempts_by_operator = defaultdict(list)
        accepted_by_operator = defaultdict(list)
        for event in attempts:
            attempts_by_operator[event.get('operator', 'unknown')].append(event)
        for event in accepted:
            accepted_by_operator[event.get('operator', 'unknown')].append(event)

        result = {}
        for operator, operator_attempts in attempts_by_operator.items():
            operator_accepted = accepted_by_operator[operator]
            deltas = [event['score_delta'] for event in operator_accepted
                      if event.get('score_delta') is not None]
            result[operator] = {
                'attempts': len(operator_attempts),
                'valid_rate': len(operator_accepted) / len(operator_attempts),
                'improvement_rate': (
                    sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0
                ),
                'mean_gain': sum(deltas) / len(deltas) if deltas else None,
                'best_gain': max(deltas) if deltas else None,
            }
        return result

    @staticmethod
    def _append_jsonl(path, record):
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

    @staticmethod
    def _load_jsonl(path):
        if not os.path.exists(path):
            return []
        with open(path, encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]
