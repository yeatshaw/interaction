from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SearchDynamicsTest(unittest.TestCase):
    def test_equal_scores_do_not_divide_by_zero_and_depth_increments(self):
        module = load_module('mcts_under_test', 'mcts.py')
        tree = module.MCTS('Root', 0.5, 0.1)
        parent = module.MCTSNode('parent', 'parent', 0, parent=tree.root, depth=1, visit=1, Q=1.0)
        tree.root.add_child(parent)
        tree.backpropagate(parent)
        child = module.MCTSNode(
            'child', 'child', 0, parent=parent, depth=parent.depth + 1,
            visit=1, Q=1.0, operator='m2', round_id=1,
        )
        self.assertEqual(child.depth, 2)
        self.assertGreaterEqual(tree.uct(parent, 1.0), 0.0)

    def test_round_state_and_offline_summary(self):
        mcts_module = load_module('mcts_for_dynamics_test', 'mcts.py')
        dynamics_module = load_module('dynamics_under_test', 'dynamics.py')
        with tempfile.TemporaryDirectory() as directory:
            recorder = dynamics_module.SearchDynamicsRecorder(directory)
            tree = mcts_module.MCTS('Root', 0.5, 0.1)
            parent = mcts_module.MCTSNode('p', 'p', 0, parent=tree.root, depth=1, visit=2, Q=1.0)
            tree.root.add_child(parent)
            tree.backpropagate(parent)
            child = mcts_module.MCTSNode('c', 'c', 0, parent=parent, depth=2, visit=1, Q=1.2)
            parent.add_child(child)
            tree.backpropagate(child)

            recorder.start_round()
            recorder.record_event({'event_type': 'attempt', 'operator': 'm2', 'status': 'evaluated'})
            recorder.record_event({
                'event_type': 'outcome', 'operator': 'm2', 'accepted': True,
                'duplicate': False, 'score_delta': 0.2,
            })
            state = recorder.record_state(tree, sample_count=1, evaluation_count=1)
            self.assertEqual(state['tree_size'], 2)
            self.assertEqual(state['max_depth'], 2)
            self.assertEqual(state['improvement_rate'], 1.0)

            result = subprocess.run(
                [sys.executable, str(HERE / 'analyze_dynamics.py'), directory],
                check=True, capture_output=True, text=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary['rounds'], 1)
            self.assertEqual(summary['operators']['m2']['improvement_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
