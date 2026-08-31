"""Summarize run_0 training results for the 59-config VRPTW experiment."""
import csv
import json
import sys
from pathlib import Path

ATTR = ('good', 'bad', 'both', 'difference', 'same')
SUMM = ('guidance', 'experience', 'conditions')


def config_label(i):
    if i < 12:
        parent = ('none', 'identical', 'shared', 'identical_shared')[i // 3]
        population = ('elite_worst', 'elite_average', 'worst_average')[i % 3]
        return f'{parent}+{population}', 'good/bad'
    j = i - 12
    # Same order as run_59 scripts: 47 non-empty method combinations.
    for comp in (0, 1):
        for attr in (0, 1):
            for summ in (0, 1):
                if not (comp or attr or summ):
                    continue
                attrs = ATTR if attr else ('good',)
                sums = SUMM if summ else ('guidance',)
                for a in attrs:
                    for s in sums:
                        if j == 0:
                            methods = '+'.join(x for x, enabled in (
                                ('comparison', comp), ('attribution', attr), ('summarization', summ)) if enabled)
                            return 'identical_shared', f'{methods}:{a if attr else "-"}:{s if summ else "-"}'
                        j -= 1
    raise ValueError(f'Unknown configuration {i}')


def read_score(path):
    files = list(path.rglob('samples_best.json'))
    if not files:
        return None
    try:
        records = json.loads(files[0].read_text(encoding='utf-8'))
        return records[-1].get('score') if records else None
    except (OSError, ValueError, TypeError, IndexError):
        return None


def main(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / 'training_run0_summary.csv'
    rows = []
    for i in range(59):
        directory = root / f'{i:02d}'
        score = read_score(directory)
        input_cfg, attribution = config_label(i)
        rows.append({'config_id': f'{i:02d}', 'input_config': input_cfg,
                     'attribution_method': attribution, 'run': 0,
                     'best_training_score': score})
    # Matrix layout: horizontal axis=input configuration, vertical axis=attribution method.
    configs = []
    for row in rows:
        if row['input_config'] not in configs:
            configs.append(row['input_config'])
    methods = []
    for row in rows:
        if row['attribution_method'] not in methods:
            methods.append(row['attribution_method'])
    values = {(row['attribution_method'], row['input_config']): row['best_training_score']
              for row in rows}
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['attribution_method'] + configs)
        for method in methods:
            writer.writerow([method] + [values.get((method, config), '') for config in configs])
    print(f'Wrote matrix ({len(methods)} rows x {len(configs)} columns) to {output}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python summarize_training_run0.py <logs/vrptw_59> [output.csv]')
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
