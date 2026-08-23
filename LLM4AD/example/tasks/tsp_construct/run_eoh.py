import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH


def main():
    log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh_tsp'))
    lineage_path = os.environ.get('LLM4AD_EOH_LINEAGE', str(log_dir / 'eoh_lineage.json'))
    llm = HttpsApi(
        host='api.apilio.ai',
        key='',
        model='gpt-4o-mini',
        timeout=60
    )

    task = TSPEvaluation(
        timeout_seconds=600,
        dataset_path="/public/home/liuyang/dataset/tsp_instances.pkl"
    )
    reflection_mode = 1

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir=str(log_dir), log_style='complex'),
                 evaluation=task,
                 max_sample_nums=1000,
                 max_generations=1000,
                 pop_size=20,
                 num_samplers=10,
                 num_evaluators=10,
                 reflection_input_mode=reflection_mode,
                 lineage_log_path=lineage_path,
                 debug_mode=False)
    method.run()

if __name__ == '__main__':
    main()