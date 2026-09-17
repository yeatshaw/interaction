"""Editable Recipe-MCTS configuration for the 1D bin-packing task."""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "dataset"

MCTS_RECIPE_CONFIG = {
    "log_dir": "logs/mcts_recipe_bp_1d",
    "checkpoint": None,
    "mode": "reflection",
    "operators": ("e1", "e2", "m1", "m2"),
    "mcts": {
        "pop_size": 10,
        "selection_num": 2,
        "max_depth": 50,
        "max_sample_count": 10000,
        "initialization_mode": "eoh",
        "initial_sample_nums_max": None,
        "init_pop_size": 20,
        "num_samplers": 10,
        "num_evaluators": 10,
        "exploration_constant": 0.4,
        "depth_balance_weight": 0.4,
        "node_batch_size": 10,
        "seed": None,
        "elite_pool_size": 0,
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
        "base_url": "https://api.apilio.ai/v1",
        "api_key": "",
        "api_key_env": ["LLM4AD_EMBEDDING_API_KEY", "OPENAI_API_KEY8", "LLM4AD_API_KEY"],
        "model": "text-embedding-v4",
        "encoding_format": "float",
        "top_k": 3,
    },
    "evaluation": {
        "timeout_seconds": 300,
        "n_instance": 8,
        "n_items": 500,
        "n_bins": 500,
        "bin_capacity": 100,
        "dataset_path": str(DATASET_DIR / "bp_1d_10k_C100_train.pkl"),
    },
    "test": {
        "enabled": True,
        "task": "bp_1d",
        "data_paths": [
            str(DATASET_DIR),
        ],
        "output": "bp_1d_test_results",
        "max_nodes": 0,
        "instance_timeout": 1800,
    },
}
