"""Run MCTS-AHD without reflection while recording search dynamics."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm4ad.method.mcts_ahd import MCTS_AHD
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase


def main():
    api_key = os.environ.get('LLM4AD_API_KEY')
    if not api_key:
        raise RuntimeError('Set LLM4AD_API_KEY before running this experiment.')

    log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/mcts_ahd_dynamics'))
    llm = HttpsApi(
        host=os.environ.get('LLM4AD_API_HOST', 'api.openai.com'),
        key=api_key,
        model=os.environ.get('LLM4AD_MODEL', 'gpt-4o-mini'),
        timeout=int(os.environ.get('LLM4AD_TIMEOUT', '60')),
    )
    method = MCTS_AHD(
        llm=llm,
        evaluation=TSPEvaluation(),
        profiler=ProfilerBase(log_dir=str(log_dir), log_style='complex'),
        dynamics_log_dir=str(log_dir / 'dynamics'),
        max_sample_nums=int(os.environ.get('LLM4AD_MAX_SAMPLES', '100')),
        init_size=4,
        pop_size=10,
        num_samplers=1,
        num_evaluators=1,
        alpha=0.5,
        lambda_0=0.1,
    )
    method.run()


if __name__ == '__main__':
    main()
