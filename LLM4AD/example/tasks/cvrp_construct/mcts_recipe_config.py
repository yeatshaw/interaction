"""Editable Recipe-MCTS configuration for the CVRP task."""

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent

MCTS_RECIPE_CONFIG = {
    "log_dir": "logs/mcts_recipe_cvrp",
    "checkpoint": None,
    "mode": "reflection",
    "operators": ("e1", "e2", "m1", "m2"),
    "mcts": {
        "pop_size": 10,
        "selection_num": 2,
        "max_depth": 50,
        "max_sample_count": 10000,
        "initialization_mode": "refineevo",
        "initial_sample_nums_max": None,
        "init_pop_size": 30,
        "num_samplers": 10,
        "num_evaluators": 10,
        "exploration_constant": 0.1,
        "depth_balance_weight": 0.2,
        "node_batch_size": 100,
        "seed": None,
        "elite_pool_size": 0,
        "debug": False,
    },
    "adaptive": {
        "enabled": False,
        "recipes_per_expansion": 3,
        "prior_precision": 1.0,
        "posterior_scale": 0.5,
        "reward_best_weight": 0.8,
        "reward_mean_weight": 0.2,
    },
    "llm": {
        "host": "api.apilio.ai",
        "api_key": "",
        "api_key_env": ["LLM4AD_API_KEY"],
        "model": "gpt-4o-mini",
        "timeout": 60,
    },
    "embedding": {
        "base_url": "https://api.apilio.ai/v1",
        "api_key": "",
        "api_key_env": ["LLM4AD_EMBEDDING_API_KEY", "OPENAI_API_KEY8", "LLM4AD_API_KEY"],
        "model": "text-embedding-v4",
        "encoding_format": "float",
        "top_k": 3,
    },
    "evaluation": {
        "timeout_seconds": 300,
        "problem_size": 100,
        "n_instance": 50,
        "capacity": 40,
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
