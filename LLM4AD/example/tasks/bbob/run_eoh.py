import sys
from datetime import datetime
from pathlib import Path

# Always resolve imports from the LLM4AD project root, independent of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.bbob.evaluation import BBOBEvaluationINI
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH, EoHProfiler
from utils import get_info
from operator_statistics_report import main as build_statistics_report


def main():
    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='xxx',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=120)
    method_name = 'mutate'
    info = get_info(method_name)
    run_name = f'{datetime.now():%Y%m%d_%H%M%S}_{method_name}'
    task = BBOBEvaluationINI(
        method_name=method_name,
        statistics_dir=Path(__file__).resolve().parent / 'operator_statistics_results' / run_name,
    )

    method = EoH(llm=llm,
                 profiler=EoHProfiler(log_dir='logs/eoh', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,
                 max_generations=5,
                 pop_size=4,
                 num_samplers=4,
                 num_evaluators=4,
                 info=info,
                 debug_mode=False)

    method.run()
    build_statistics_report(task.statistics_dir)

if __name__ == '__main__':
    main()
