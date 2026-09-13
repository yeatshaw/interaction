"""Summarize population-node counts and best-score distributions by depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_population_nodes(experiment_dir: str | Path) -> list[dict]:
    """Load and deduplicate nodes from population_nodes_*.json files."""
    experiment_dir = Path(experiment_dir)
    files = sorted(experiment_dir.glob("population_nodes_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No population_nodes_*.json files found in {experiment_dir}")

    nodes_by_id = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", payload) if isinstance(payload, dict) else payload
        if not isinstance(nodes, list):
            raise ValueError(f"Invalid node payload in {path}")
        for node in nodes:
            node_id = node.get("population_node_id")
            if node_id is None:
                raise ValueError(f"Node without population_node_id in {path}")
            nodes_by_id[int(node_id)] = node
    return sorted(nodes_by_id.values(), key=lambda node: int(node["population_node_id"]))


def summarize_nodes_by_depth(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Write per-depth statistics for population-node ``best_score`` values."""
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir) if output_dir else experiment_dir / "tree_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = load_population_nodes(experiment_dir)
    saved_node_ids = {int(node["population_node_id"]) for node in nodes}
    missing_best_node_ids = []
    sample_best_path = experiment_dir / "sample_best.json"
    if sample_best_path.is_file():
        best_records = json.loads(sample_best_path.read_text(encoding="utf-8"))
        if isinstance(best_records, dict):
            best_records = [best_records]
        missing_best_node_ids = sorted({
            int(record["population_node_id"])
            for record in best_records
            if record.get("population_node_id") is not None
            and int(record["population_node_id"]) not in saved_node_ids
        })
    depth_scores: dict[int, list[float]] = {}
    detail_rows = []
    for node in nodes:
        depth = int(node["depth"])
        score = node.get("best_score")
        if score is None or not np.isfinite(float(score)):
            continue
        score = float(score)
        depth_scores.setdefault(depth, []).append(score)
        detail_rows.append({
            "population_node_id": int(node["population_node_id"]),
            "parent_population_node_id": node.get("parent_population_node_id"),
            "depth": depth,
            "recipe_id": node.get("incoming_recipe_id"),
            "best_score": score,
            "population_size": node.get("population_size"),
        })

    if not depth_scores:
        raise ValueError(f"No finite best_score values found in {experiment_dir}")

    summary = []
    for depth in sorted(depth_scores):
        values = np.asarray(depth_scores[depth], dtype=float)
        summary.append({
            "depth": depth,
            "node_count": len(values),
            "unique_score_count": len(np.unique(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        })

    summary_path = output_dir / "depth_score_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    detail_path = output_dir / "population_node_scores.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    try:
        import matplotlib.pyplot as plt

        depths = sorted(depth_scores)
        figure_width = max(10.0, min(24.0, len(depths) * 0.55))
        fig, ax = plt.subplots(figsize=(figure_width, 6.5))
        ax.boxplot([depth_scores[depth] for depth in depths], tick_labels=depths,
                   showmeans=True, showfliers=True)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Best score")
        ax.set_title("Population-node best-score distribution by depth")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "depth_score_boxplot.png", dpi=200)
        plt.close(fig)
    except (ImportError, TypeError):
        # Older matplotlib uses ``labels`` instead of ``tick_labels``. CSV
        # outputs remain the authoritative result if plotting is unavailable.
        pass

    print(f"nodes={len(detail_rows)} depths={len(summary)}")
    print(f"summary={summary_path}")
    print(f"details={detail_path}")
    if missing_best_node_ids:
        print(
            "WARNING: sample_best.json references population nodes not yet "
            "present in population_nodes files: "
            + ",".join(map(str, missing_best_node_ids))
        )
        print("The depth summary is incomplete until those nodes are flushed.")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Recipe-MCTS population nodes by tree depth.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_nodes_by_depth(args.experiment_dir, args.output_dir)
