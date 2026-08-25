import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm4ad.task.optimization.vrptw_construct import VRPTWEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from example.tasks.utils import get_info


def main():
    base_log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh_vrptw_非反思算子'))
    info = get_info('select_next_node', 'llm4ad.task.optimization.vrptw_construct.template')

    # Four bits are, in order: parent, best/worst, population average,
    # reflection guidance.  Thus 1000 enables parent information only.
    run_log_dir = base_log_dir
    llm = HttpsApi(
        host='api.apilio.ai',
        key='',
        model='gpt-4o-mini',
        timeout=60
    )
    task = VRPTWEvaluation(
        timeout_seconds=300,
        dataset_path='/public/home/liuyang/dataset/vrptw_instances.pkl',
        n_instance=100,
        instance_workers=10
    )
    method = EoH(
                llm=llm,
                profiler=EoHProfiler(log_dir=str(run_log_dir), log_style='complex'),
                evaluation=task,
                max_sample_nums=500,
                max_generations=500,
                pop_size=10,
                num_samplers=10,
                num_evaluators=10,
                reflection_parent_info=True,
                reflection_best_worst=True,
                reflection_fitness=2,
                reflection_avg_fitness=True,
                reflection_check_guidance=True,
                lineage_log_path='eoh_lineage.json',
                info=info,
                debug_mode=False
            )
    method.run()


if __name__ == '__main__':
    main()
