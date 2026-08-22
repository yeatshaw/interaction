from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(log_dir):
    events = load_jsonl(log_dir / 'expansion_events.jsonl')
    states = load_jsonl(log_dir / 'search_states.jsonl')
    attempts = [event for event in events if event.get('event_type') == 'attempt']
    outcomes = [event for event in events if event.get('event_type') == 'outcome']
    accepted = [event for event in outcomes if event.get('accepted')]
    per_operator = defaultdict(lambda: {'attempts': [], 'accepted': []})
    for event in attempts:
        per_operator[event.get('operator', 'unknown')]['attempts'].append(event)
    for event in accepted:
        per_operator[event.get('operator', 'unknown')]['accepted'].append(event)

    operators = {}
    for operator, rows in sorted(per_operator.items()):
        deltas = [event['score_delta'] for event in rows['accepted']
                  if event.get('score_delta') is not None]
        operators[operator] = {
            'attempts': len(rows['attempts']),
            'valid_rate': len(rows['accepted']) / len(rows['attempts']) if rows['attempts'] else 0.0,
            'improvement_rate': sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0,
            'mean_gain': sum(deltas) / len(deltas) if deltas else None,
            'best_gain': max(deltas) if deltas else None,
        }

    return {
        'rounds': len(states),
        'samples': len(attempts),
        'accepted_expansions': len(accepted),
        'final_best_score': states[-1]['best_score'] if states else None,
        'final_tree_size': states[-1]['tree_size'] if states else 0,
        'final_stagnation_length': states[-1]['stagnation_length'] if states else 0,
        'mean_root_visit_entropy': (
            sum(state['root_visit_entropy'] for state in states) / len(states) if states else None
        ),
        'operators': operators,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize MCTS-AHD search-dynamics logs.')
    parser.add_argument('log_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log_dir), ensure_ascii=False, indent=2))
