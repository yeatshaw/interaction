from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from llm4ad.task.optimization.vrptw_construct import VRPTWEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from example.tasks.mcts_recipe_runner import common_options, run_task


def main():
    options = common_options("logs/mcts_recipe_vrptw")
    evaluation = VRPTWEvaluation(
        timeout_seconds=int(os.environ.get('LLM4AD_TIMEOUT', '300')),
        problem_size=int(os.environ.get('LLM4AD_PROBLEM_SIZE', '100')),
        n_instance=int(os.environ.get('LLM4AD_N_INSTANCE', '50')),
        dataset_path=os.environ.get('LLM4AD_VRPTW_TRAIN_DATA'),
    )
    llm = HttpsApi(
        host=os.environ.get('LLM4AD_API_HOST', 'api.apilio.ai'),
        key=os.environ.get('LLM4AD_API_KEY', ''),
        model=os.environ.get('LLM4AD_API_MODEL', 'gpt-4o-mini'),
        timeout=int(os.environ.get('LLM4AD_API_TIMEOUT', '60')),
    )
    run_task(method_name='select_next_node',
             template_module='llm4ad.task.optimization.vrptw_construct.template',
             evaluation=evaluation, llm=llm, **options)


if __name__ == '__main__':
    main()
