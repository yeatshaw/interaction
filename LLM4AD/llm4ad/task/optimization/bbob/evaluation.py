

from __future__ import annotations
from typing import Callable, Any

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bbob.core import *

__all__ = ['BBOBEvaluationINI']


class BBOBEvaluationINI(Evaluation):

    def __init__(self,
                 timeout_seconds: int =600,
                 method_name: str = 'mutate',
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

        # Get the current script's directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to the folder where you want to read/write files
        self.folder_path = self.current_dir


    def evaluate(self, eva: Callable, program_str: str) -> float:

        fitness = main(num_process=None, total_runs=None, test_problems=None,
                       random_seed=None, func=program_str, method_name=self.method_name)

        return -fitness  # Negative because we want to minimize the number of bins

    def evaluate_program(self, program_str: str, callable_func: Callable, **kwargs) -> Any | None:
        return self.evaluate(callable_func, program_str)
