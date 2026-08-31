import os
from pathlib import Path
import sys
import json
import pickle
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


def evaluate_best_on_solomon(log_dir, dataset_path):
    """Evaluate the final best program separately on every named Solomon instance."""
    best_path = Path(log_dir) / 'samples' / 'samples_best.json'
    if not best_path.exists():
        return
    with best_path.open(encoding='utf-8') as f:
        records = json.load(f)
    if not records:
        return
    program = records[-1].get('program') or records[-1].get('function')
    if not program:
        return
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    evaluator = VRPTWEvaluation(dataset_path=None, n_instance=1, problem_size=100)
    results = {}
    for name, data in datasets.items():
        evaluator.problem_size = len(data[0]) - 1
        try:
            value = evaluator._evaluate_instance(program, data)
            results[name] = None if value is None or value == float('inf') else float(value)
        except Exception as exc:
            results[name] = {'error': str(exc)}
    with (Path(log_dir) / 'solomon_per_instance.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    base_log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh'))
    info = get_info('select_next_node', 'llm4ad.task.optimization.vrptw_construct.template')

    # Main reflection ablation settings.  The three behaviors are independent
    # and can be enabled together.  Environment variables override defaults.
    # 1/0: enable/disable comparison, attribution, and summary instructions.
    reflection_comparison_flag = os.environ.get('LLM4AD_REFLECTION_COMPARISON', '1') == '1'
    reflection_attribution_flag = os.environ.get('LLM4AD_REFLECTION_ATTRIBUTION', '0') == '1'
    reflection_summarization_flag = os.environ.get('LLM4AD_REFLECTION_SUMMARY', '0') == '1'
    # Attribution task: good, bad, both, difference, or same.
    reflection_attribution = os.environ.get('LLM4AD_ATTRIBUTION_TYPE', 'good')
    # Summary task: guidance, experience, or conditions.
    reflection_summarization = os.environ.get('LLM4AD_SUMMARY_TYPE', 'guidance')

    # Comparison input ablation.  The two flags form four combinations:
    # neither, identical parent sets, at least one shared parent, or both.
    # Population comparison: elite_worst, elite_average, worst_average, or empty.
    population_comparison = os.environ.get('LLM4AD_POPULATION_COMPARISON', '') or None
    # 1/0: require selected children to have identical complete parent sets.
    identical_parent_children = os.environ.get('LLM4AD_IDENTICAL_PARENTS', '0') == '1'
    # 1/0: require selected children to share at least one parent.
    shared_parent_children = os.environ.get('LLM4AD_SHARED_PARENT', '0') == '1'

    # Two-bit input setting: population average and reflection guidance.
    # Parent and elite/worst inputs are controlled by the dedicated settings above.
    # Two bits, each 0/1: [population average score, prior suggestion/experience].
    bits = os.environ.get('LLM4AD_REFLECTION_BITS', '11')
    if len(bits) != 2 or any(bit not in '01' for bit in bits):
        raise ValueError('LLM4AD_REFLECTION_BITS must be a two-bit string, e.g. 01.')
    runs = [int(os.environ['LLM4AD_RUN_INDEX'])] if os.environ.get('LLM4AD_RUN_INDEX') else range(3)
    fitness_values = ([int(os.environ['LLM4AD_REFLECTION_FITNESS'])]
                      if os.environ.get('LLM4AD_REFLECTION_FITNESS') else range(3))
    for run in runs:
        for reflection_fitness in fitness_values:
            reflection_parent_info = False
            reflection_best_worst = False
            reflection_avg_fitness = bits[0] == '1'
            reflection_check_guidance = bits[1] == '1'
            run_log_dir = base_log_dir / str(reflection_fitness) / bits / str(run)
            llm = HttpsApi(
                host='api.apilio.ai', key='', model='gpt-4o-mini', timeout=60
            )
            task = VRPTWEvaluation(
                    timeout_seconds=300,
                    dataset_path='/public/home/liuyang/dataset/vrptw_instances.pkl',
                    n_instance=100,
                    instance_workers=1
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
                            reflection_identical_parent_children=identical_parent_children,
                            reflection_shared_parent_children=shared_parent_children,
                            reflection_population_comparison=population_comparison,
                            reflection_comparison_flag=reflection_comparison_flag,
                            reflection_attribution_flag=reflection_attribution_flag,
                            reflection_summarization_flag=reflection_summarization_flag,
                            reflection_attribution=reflection_attribution,
                            reflection_summarization=reflection_summarization,
                            use_long_term_reflection=False,
                            lineage_log_path='eoh_lineage.json',
                            info=info,
                            debug_mode=False
                    )
            method.run()
            save_convergence_plot(method._profiler._log_dir, method._convergence_history)
            solomon_path = os.environ.get('LLM4AD_SOLOMON_DATASET')
            if solomon_path:
                evaluate_best_on_solomon(method._profiler._log_dir, solomon_path)


if __name__ == '__main__':
    main()
