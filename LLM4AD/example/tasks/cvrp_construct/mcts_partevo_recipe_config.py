"""Editable configuration for the CVRP MCTS-PartEvo-Recipe experiment."""

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent

MCTS_PARTEVO_RECIPE_CONFIG = {
    "task": "cvrp",
    "log_dir": "logs/mcts_partevo_recipe_cvrp",
    "checkpoint": None,
    "recipes": None,
    "mcts_partevo": {
        "pop_size": 16,
        "partition_number": 4,
        "operators": ("re", "se", "cn", "lge"),
        "operators_parent_num": {"re": 1, "se": 1, "cn": 1, "lge": 1},
        "operators_frequency": {"re": 1, "se": 1, "cn": 1, "lge": 1},
        "max_depth": 50,
        "max_sample_count": 5000,
        "num_samplers": 16,
        "num_evaluators": 16,
        "initial_num_samplers": 16,
        "init_pop_size": 30,
        "max_round_attempts": 3,
        "exploration_constant": 0.2,
        "depth_balance_weight": 0.4,
        "node_batch_size": 100,
        "feature_used": ("ast",),
        "use_resource_tilt": False,
        "bert_model_path": None,
        "seed": None,
        "debug": False,
        "adaptive": {
            "enabled": False,
            "strategy": "linear_thompson_sampling",
            "recipes_per_expansion": 3,
            "prior_precision": 1.0,
            "posterior_scale": 0.5,
            "discount_factor": 0.9,
            "window_size": 50,
            "exploration": 1.0,
        },
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
