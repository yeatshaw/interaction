import os
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm4ad.task.optimization.vrptw_construct import VRPTWEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from example.tasks.utils import get_info


def save_convergence_plot(log_dir, convergence_history):
    if not convergence_history:
        return
    sample_ids, best_scores = zip(*convergence_history)
    output = Path(log_dir) / 'convergence.png'
    plt.figure(figsize=(8, 5))
    plt.plot(sample_ids, best_scores, linewidth=1.8)
    plt.xlabel('ID')
    plt.ylabel('Best score so far')
    plt.title('VRPTW Search Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main():
    base_log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh_vrptw_非反思算子'))
    info = get_info('select_next_node', 'llm4ad.task.optimization.vrptw_construct.template')
    
    # Four bits are, in order: parent, best/worst, population average,
    # reflection guidance.  Thus 1000 enables parent information only.
    bits = os.environ.get('LLM4AD_REFLECTION_BITS', '1111')
    for run in range(3):
        for reflection_fitness in range(3):
            reflection_parent_info = bits[0] == '1'
            reflection_best_worst = bits[1] == '1'
            reflection_avg_fitness = bits[2] == '1'
            reflection_check_guidance = bits[3] == '1'
            run_log_dir = base_log_dir / str(reflection_fitness) / bits / str(run)
            llm = HttpsApi(
                host='api.apilio.ai', key='', model='gpt-4o-mini', timeout=60
            )
            task = VRPTWEvaluation(
                    timeout_seconds=300,
                    dataset_path='/public/home/liuyang/dataset/vrptw_instances.pkl',
                    n_instance=100,
                    instance_workers=10
            )
            print(f'Running reflection_fitness={reflection_fitness}, combination={bits}, run={run}')
            method = EoH(
                        llm=llm,
                        profiler=EoHProfiler(log_dir=str(run_log_dir), log_style='complex'),
                            evaluation=task,
                            max_sample_nums=500,
                            max_generations=500,
                            pop_size=10,
                            num_samplers=10,
                            num_evaluators=10,
                            reflection_parent_info=reflection_parent_info,
                            reflection_best_worst=reflection_best_worst,
                            reflection_fitness=reflection_fitness,
                            reflection_avg_fitness=reflection_avg_fitness,
                            reflection_check_guidance=reflection_check_guidance,
                            use_long_term_reflection=(
                                os.environ.get('LLM4AD_LONG_TERM_REFLECTION', '1') == '1'
                            ),
                            lineage_log_path='eoh_lineage.json',
                            info=info,
                            debug_mode=False
                    )
            method.run()
            save_convergence_plot(method._profiler._log_dir, method._convergence_history)


if __name__ == '__main__':
    main()
