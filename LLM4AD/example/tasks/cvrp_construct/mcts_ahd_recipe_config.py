"""Editable MCTS-AHD-Recipe configuration for constructive CVRP."""

from pathlib import Path

from llm4ad.method.mcts_recipe.recipes import get_default_recipes


TASK_DIR = Path(__file__).resolve().parent

MCTS_AHD_RECIPE_CONFIG = {
    "log_dir": "logs/mcts_ahd_recipe_cvrp",
    "state_dir": "logs/mcts_ahd_recipe_cvrp/mcts_ahd_recipe_state",
    "resume_mode": False,
    "recipes": get_default_recipes(),
    "method": {
        "max_sample_nums": 5000,
        "init_size": 10,
        "pop_size": 10,
        "selection_num": 2,
        "num_samplers": 10,
        "num_evaluators": 10,
        "multi_thread_or_process_eval": "thread",
        "alpha": 0.5,
        "lambda_0": 0.1,
        "recipe_temperature": 1.0,
        "recipe_batch_expansion": True,
        "recipes_per_node": 5,
        "expansion_workers": 30,
        "node_batch_size": 100,
        "log_style": "complex",
        "debug": False,
    },
    "llm": {
        "host": "api.apilio.ai",
        "api_key": "",
        "api_key_env": ["LLM4AD_API_KEY"],
        "model": "gpt-4o-mini",
        "timeout": 60,
    },
    "embedding": {
        "enabled": True,
        "base_url": "https://api.apilio.ai/v1",
        "api_key": "",
        "api_key_env": ["LLM4AD_EMBEDDING_API_KEY", "LLM4AD_API_KEY"],
        "model": "text-embedding-v4",
        "encoding_format": "float",
    },
    "evaluation": {
        "timeout_seconds": 300,
        "problem_size": 100,
        "n_instance": 50,
        "capacity": 50,
        "dataset_path": str(TASK_DIR / "cvrp_instances_train_n100c50.pkl"),
    },
    "test": {
        "enabled": True,
        "task": "cvrp",
        "data_paths": [str(TASK_DIR / "cvrp_instances_test_n200c50.pkl")],
        "output": "cvrp_test_results.csv",
        "max_nodes": 0,
        "instance_timeout": 1800,
    },
}
