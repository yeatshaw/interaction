import csv
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from example.tasks.utils import get_info
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.task.optimization.fssp_construct import FSSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def _best_program(log_dir):
    path = Path(log_dir) / 'samples' / 'samples_best.json'
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding='utf-8'))
    return (records[-1].get('program') or records[-1].get('function')) if records else None


def _load_object_npz(path, key):
    """Read object NPZ files created by NumPy 2.x with NumPy 1.x."""
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = importlib.import_module('numpy.core')
        for module in ('multiarray', 'numeric', '_multiarray_umath'):
            try:
                sys.modules[f'numpy._core.{module}'] = importlib.import_module(
                    f'numpy.core.{module}')
            except ImportError:
                pass
    with np.load(path, allow_pickle=True) as loaded:
        return loaded[key].item()


def evaluate_taillard(log_dir, dataset_path):
    """Load Taillard only after training and preserve file/instance ordering."""
    program = _best_program(log_dir)
    if not program:
        return
    datasets = _load_object_npz(dataset_path, 'fsp_dict')
    evaluator = FSSPEvaluation(n_instance=1, n_jobs=1, n_machines=1)
    rows = []
    for file_name in sorted(datasets):
        for instance_id, (jobs, machines, best_known, processing_times) in enumerate(datasets[file_name], 1):
            try:
                makespan = evaluator.evaluate_instance(program, np.asarray(processing_times))
            except Exception:
                makespan = None
            gap = ((makespan - best_known) / best_known
                   if makespan is not None and best_known else None)
            rows.append((file_name, instance_id, jobs, machines,
                         makespan, best_known, gap))
    with (Path(log_dir) / 'fssp_test_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'instance_id', 'jobs', 'machines',
                         'makespan', 'best_known', 'gap'])
        writer.writerows(rows)


def save_convergence_plot(log_dir, history):
    if not history:
        return
    x, y = zip(*history)
    plt.figure(figsize=(8, 5)); plt.plot(x, y)
    plt.xlabel('ID'); plt.ylabel('Best score so far'); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(Path(log_dir) / 'convergence.png', dpi=150); plt.close()


def main():
    info = get_info('select_next_job', 'llm4ad.task.optimization.fssp_construct.template')
    bits = os.environ.get('LLM4AD_REFLECTION_BITS', '11')
    if len(bits) != 2 or any(bit not in '01' for bit in bits):
        raise ValueError('LLM4AD_REFLECTION_BITS must be a two-bit string.')
    run = int(os.environ.get('LLM4AD_RUN_INDEX', '0'))
    fitness = int(os.environ.get('LLM4AD_REFLECTION_FITNESS', '2'))
    log_dir = Path(os.environ.get('LLM4AD_LOG_DIR', 'logs/eoh_fssp')) / str(fitness) / bits / str(run)
    task = FSSPEvaluation(timeout_seconds=300, n_instance=50, n_jobs=50, n_machines=20)
    method = EoH(
        llm=HttpsApi(host='api.apilio.ai', key='', model='gpt-4o-mini', timeout=60),
        profiler=EoHProfiler(log_dir=str(log_dir), log_style='complex'),
        evaluation=task, max_sample_nums=500, max_generations=500, pop_size=10,
        num_samplers=int(os.environ.get('LLM4AD_NUM_SAMPLERS', '1')),
        num_evaluators=int(os.environ.get('LLM4AD_NUM_EVALUATORS', '1')),
        reflection_parent_info=False, reflection_best_worst=False,
        reflection_fitness=fitness, reflection_avg_fitness=bits[0] == '1',
        reflection_check_guidance=bits[1] == '1',
        reflection_identical_parent_children=os.environ.get('LLM4AD_IDENTICAL_PARENTS', '0') == '1',
        reflection_shared_parent_children=os.environ.get('LLM4AD_SHARED_PARENT', '0') == '1',
        reflection_population_comparison=os.environ.get('LLM4AD_POPULATION_COMPARISON') or None,
        reflection_comparison_flag=os.environ.get('LLM4AD_REFLECTION_COMPARISON', '1') == '1',
        reflection_attribution_flag=os.environ.get('LLM4AD_REFLECTION_ATTRIBUTION', '0') == '1',
        reflection_summarization_flag=os.environ.get('LLM4AD_REFLECTION_SUMMARY', '0') == '1',
        reflection_attribution=os.environ.get('LLM4AD_ATTRIBUTION_TYPE', 'good'),
        reflection_summarization=os.environ.get('LLM4AD_SUMMARY_TYPE', 'guidance'),
        use_long_term_reflection=os.environ.get('LLM4AD_LONG_TERM_REFLECTION', '0') == '1',
        lineage_log_path='eoh_lineage.json', info=info, debug_mode=False)
    method.run()
    save_convergence_plot(method._profiler._log_dir, method._convergence_history)
    test_path = os.environ.get('LLM4AD_FSSP_TEST_DATA')
    if test_path:
        evaluate_taillard(method._profiler._log_dir, test_path)


if __name__ == '__main__':
    main()
