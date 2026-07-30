

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Callable, Any

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bbob.core import *
from llm4ad.task.optimization.bbob.operator_statistics_plugin import instrument_program

__all__ = ['BBOBEvaluationINI']


class BBOBEvaluationINI(Evaluation):

    def __init__(self,
                 timeout_seconds: int =600,
                 method_name: str = 'mutate',
                 statistics_dir: str | None = None,
                 **kwargs):
        """
        Args:
            n_bins: The number of available bins at the beginning.
        """
        super().__init__(
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.method_name = method_name
        self.statistics_dir = statistics_dir

        # Get the current script's directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to the folder where you want to read/write files
        self.folder_path = self.current_dir


    def evaluate(self, eva: Callable, program_str: str) -> float:

        candidate_id = hashlib.sha256(program_str.encode('utf-8')).hexdigest()[:16]
        evaluated_program = (instrument_program(program_str, self.method_name, self.statistics_dir, candidate_id)
                             if self.statistics_dir else program_str)
        fitness = main(num_process=None, total_runs=None, test_problems=None,
                       random_seed=None, func=evaluated_program, method_name=self.method_name,
                       statistics_dir=str(self.statistics_dir) if self.statistics_dir else None,
                       candidate_id=candidate_id)
        if self.statistics_dir:
            score_path = Path(self.statistics_dir) / 'scores' / f'{candidate_id}.json'
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps({
                'candidate_id': candidate_id,
                'score': -fitness,
                'loss': max(0.0, float(fitness)),
            }), encoding='utf-8')

        return -fitness

    def evaluate_program(self, program_str: str, callable_func: Callable, **kwargs) -> Any | None:
        return self.evaluate(callable_func, program_str)
