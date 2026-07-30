"""Aggregate all DE-operator statistics and draw score-bin boxplots."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BIN_UPPERS = (1e-8, 1e-4, 1.0, 10.0, 100.0, 1000.0)
LABELS = ('≤1e-8', '(1e-8,1e-4]', '(1e-4,1]',
          '(1,10]', '(10,100]', '(100,1000]', '>1000')
METADATA = {'candidate_id', 'operator_name', 'process_id', 'stage'}


def _score_bin(loss: float) -> str:
    for label, upper in zip(LABELS, BIN_UPPERS):
        if loss <= upper:
            return label
    return LABELS[-1]


def main(results_dir: str = 'mutation_statistics_results') -> None:
    results = Path(results_dir)
    scores = {}
    for path in (results / 'scores').glob('*.json'):
        score_data = json.loads(path.read_text(encoding='utf-8'))
        candidate_id = score_data.get('candidate_id', path.stem)
        loss = score_data.get('loss')
        if loss is None and 'score' in score_data:
            loss = -float(score_data['score'])
        if loss is not None:
            scores[candidate_id] = max(0.0, float(loss))
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    raw_by_candidate_stage = defaultdict(list)
    for path in (results / 'raw').glob('*.json'):
        for row in json.loads(path.read_text(encoding='utf-8')):
            raw_by_candidate_stage[(row['candidate_id'], row['operator_name'], row['stage'])].append(row)

    raw_candidate_ids = {candidate_id for candidate_id, _, _ in raw_by_candidate_stage}
    matched_candidate_ids = raw_candidate_ids & scores.keys()
    if not matched_candidate_ids:
        raise RuntimeError(
            f'No raw/score candidate IDs match in {results}. '
            f'raw examples: {sorted(raw_candidate_ids)[:3]}; '
            f'score examples: {sorted(scores)[:3]}.'
        )

    metrics_by_operator = defaultdict(set)
    for (_, operator_name, _), repeats in raw_by_candidate_stage.items():
        for item in repeats:
            metrics_by_operator[operator_name].update(
                key for key, value in item.items()
                if key not in METADATA and not key.endswith('__weight') and isinstance(value, (int, float))
            )

    rows = []
    for (candidate_id, operator_name, stage), repeats in raw_by_candidate_stage.items():
        loss = scores.get(candidate_id)
        if loss is None or not np.isfinite(loss):
            continue
        row = {
            'candidate_id': candidate_id,
            'operator_name': operator_name,
            'stage': stage,
            'repeat_n': len(repeats),
            'loss': loss,
            'log_loss': math.log10(max(loss, 1e-12)),
            'score_bin': _score_bin(loss),
        }
        for metric in metrics_by_operator[operator_name]:
            valid_items = [item for item in repeats if np.isfinite(item.get(metric, np.nan))]
            weights = [item.get(f'{metric}__weight', 1) for item in valid_items]
            denominator = sum(weights)
            row[metric] = (sum(item[metric] * weight for item, weight in zip(valid_items, weights)) / denominator
                           if denominator else np.nan)
        rows.append(row)
        for metric in metrics_by_operator[operator_name]:
            if np.isfinite(row[metric]):
                grouped[operator_name][stage][metric][row['score_bin']].append(row[metric])
    if not rows:
        raise RuntimeError(f'No finite statistics remain after matching raw and scores in {results}.')

    with (results / 'candidate_stage_metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for operator_name, stages in grouped.items():
        for stage, metrics in stages.items():
            for metric, bins in metrics.items():
                for label, values in bins.items():
                    summary.append({'operator_name': operator_name, 'stage': stage, 'metric': metric,
                                    'score_bin': label, 'n': len(values), 'q1': np.quantile(values, .25),
                                    'median': np.median(values), 'q3': np.quantile(values, .75)})

    comparison_bins = LABELS
    for operator_name, metric_names in metrics_by_operator.items():
        metric_names = sorted(metric_names)
        fig, axes = plt.subplots(len(metric_names), len(comparison_bins),
                                 figsize=(max(16, 3.1 * len(comparison_bins)),
                                          max(4, 2.8 * len(metric_names))), squeeze=False)
        for row_index, metric in enumerate(metric_names):
            for column_index, score_bin in enumerate(comparison_bins):
                axis = axes[row_index, column_index]
                data, positions = [], []
                for stage in range(10):
                    values = grouped[operator_name][stage][metric][score_bin]
                    if values:
                        data.append(values)
                        positions.append(stage + 1)
                if data:
                    axis.boxplot(data, positions=positions, widths=.55, showfliers=False)
                else:
                    axis.text(.5, .5, 'no finite samples', ha='center', va='center', transform=axis.transAxes)
                axis.set_xticks(range(1, 11), [f'{start}-{start + 10}' for start in range(0, 100, 10)], rotation=35)
                if row_index == 0:
                    axis.set_title(f'final loss {score_bin}')
                if column_index == 0:
                    axis.set_ylabel(metric)
        fig.suptitle(f'{operator_name} statistics', y=1.01)
        fig.supxlabel('budget progress (%)')
        fig.tight_layout()
        fig.savefig(results / f'boxplot_{operator_name}_metrics_by_stage.png', dpi=180)
        plt.close(fig)

    with (results / 'score_bin_summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mutation_statistics_results')
