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


def save_convergence_plot(log_dir):
    samples_dir = Path(log_dir) / 'samples'
    records = []
    for path in sorted(samples_dir.glob('samples_*.json')):
        if path.name == 'samples_best.json':
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict))
    records = [r for r in records if r.get('sample_order') is not None
               and isinstance(r.get('score'), (int, float))]
    records.sort(key=lambda r: r['sample_order'])
    if not records:
        return
    sample_ids = [r['sample_order'] for r in records]
    best_scores = []
    best = float('-inf')
    for record in records:
        best = max(best, record['score'])
        best_scores.append(best)
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
    for run in range(3):
        for reflection_fitness in range(3):
            for combo in range(16):
                bits = f'{combo:04b}'
                reflection_parent_info = bits[0] == '1'
                reflection_best_worst = bits[1] == '1'
                reflection_avg_fitness = bits[2] == '1'
                reflection_check_guidance = bits[3] == '1'
                run_log_dir = base_log_dir / str(reflection_fitness) / bits / str(run)
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
                            lineage_log_path='eoh_lineage.json',
                            info=info,
                            debug_mode=False
                        )
                method.run()
                save_convergence_plot(method._profiler._log_dir)


if __name__ == '__main__':
    main()
