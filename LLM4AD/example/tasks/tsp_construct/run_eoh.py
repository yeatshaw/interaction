import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from example.tasks.utils import get_info


def main():
    log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh_tsp'))
    # EoH places the default lineage file in the profiler's timestamped run
    # directory. An explicit path remains available for resume/debug runs.
    lineage_path = os.environ.get('LLM4AD_EOH_LINEAGE', 'eoh_lineage.json')
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
    info = get_info(
        'select_next_node',
        'llm4ad.task.optimization.tsp_construct.template'
    )
    reflection_mode = 1

    method = EoH(llm=llm,
                 profiler=EoHProfiler(log_dir=str(log_dir), log_style='complex'),
                 evaluation=task,
                 max_sample_nums=50,
                 max_generations=50,
                 pop_size=5,
                 num_samplers=1,
                 num_evaluators=1,
                 reflection_input_mode=reflection_mode,
                 lineage_log_path=lineage_path,
                 info=info,
                 debug_mode=True)
    method.run()

if __name__ == '__main__':
    main()
