"""Aggregate Solomon per-instance results across run_0/run_1/run_2.

Usage:
  python summarize_solomon_runs.py <results_root> [output_dir]
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _run_id(path):
    for part in reversed(path.parts):
        m = re.fullmatch(r'run[_-]([012])', part, re.I)
        if m:
            return int(m.group(1))
    m = re.search(r'(?:run|_)([012])(?:\D|$)', path.stem, re.I)
    return int(m.group(1)) if m else None


def _labels(path):
    text = '/'.join(path.parts).lower()
    attribution = next((x for x in ('good', 'bad', 'both', 'difference', 'same') if x in text), 'unknown')
    # Prefer an explicit input label; otherwise retain the configuration directory name.
    input_cfg = next((x for x in ('reference', 'parent_child', 'identical', 'shared', 'elite_worst', 'elite_average', 'worst_average') if x in text), None)
    if input_cfg is None:
        input_cfg = path.parent.name
    return input_cfg, attribution


def aggregate(root, output_dir=None):
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(dict)
    for result_path in root.rglob('solomon_per_instance.json'):
        run = _run_id(result_path)
        if run is None:
            # Also support layouts where each run is a direct result file under a run directory.
            continue
        input_cfg, attribution = _labels(result_path)
        data = json.loads(result_path.read_text(encoding='utf-8'))
        for instance, value in data.items():
            if isinstance(value, (int, float)) and value is not None:
                groups[(input_cfg, attribution, instance)][run] = float(value)

    rows = []
    for (input_cfg, attribution, instance), runs in sorted(groups.items()):
        rows.append({
            'input_config': input_cfg,
            'attribution_method': attribution,
            'instance': instance,
            'run_count': len(runs),
            'mean_score': sum(runs.values()) / len(runs),
        })
    csv_path = output_dir / 'solomon_runs_mean.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['input_config', 'attribution_method', 'instance', 'run_count', 'mean_score'])
        writer.writeheader(); writer.writerows(rows)
    # Matrix-style table: rows are attribution methods, columns are input configurations.
    matrix = defaultdict(dict)
    for row in rows:
        matrix[row['attribution_method'], row['instance']][row['input_config']] = row['mean_score']
    matrix_path = output_dir / 'solomon_runs_mean_matrix.csv'
    configs = sorted({r['input_config'] for r in rows})
    with matrix_path.open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['attribution_method', 'instance'] + configs
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for (method, instance), values in sorted(matrix.items()):
            writer.writerow({'attribution_method': method, 'instance': instance, **values})
    print(f'Wrote {len(rows)} instance rows to {csv_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python summarize_solomon_runs.py <results_root> [output_dir]')
    aggregate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
