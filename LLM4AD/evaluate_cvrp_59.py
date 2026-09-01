"""Evaluate the final sample_best program of 59 CVRP experiments in parallel.

Usage:
  python evaluate_cvrp_59.py --results-root <vrptw_59_run0> \
      --test-data <cvrp_test_lt200.npz> --workers 8
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def load_object_npz(path, key):
    # Files produced with NumPy 2.x may refer to numpy._core when unpickled.
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = importlib.import_module('numpy.core')
        for name in ('multiarray', 'numeric', '_multiarray_umath'):
            try:
                sys.modules[f'numpy._core.{name}'] = importlib.import_module(
                    f'numpy.core.{name}')
            except ImportError:
                pass
    with np.load(path, allow_pickle=True) as data:
        return data[key].item()


def read_best(log_dir):
    candidates = list(Path(log_dir).rglob('samples_best.json'))
    if not candidates:
        return None
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    records = json.loads(path.read_text(encoding='utf-8'))
    if not records:
        return None
    return records[-1].get('program') or records[-1].get('function')


def evaluate_one(job):
    config_id, log_dir, test_data = job
    try:
        program = read_best(log_dir)
        if not program:
            return config_id, 'missing best program'
        namespace = {'np': np}
        exec(program, namespace)
        heuristic = namespace.get('select_next_node')
        if not callable(heuristic):
            raise ValueError('best program has no callable select_next_node')

        rows = []
        for name in sorted(test_data):
            capacity, node_count, coordinates, demands, bks = test_data[name]
            distance = np.rint(np.linalg.norm(
                coordinates[:, None] - coordinates[None, :], axis=2))
            result = evaluate_instance(heuristic, distance, demands, capacity)
            gap = ((result - bks) / bks) if result is not None and bks else None
            rows.append((name, node_count - 1, result, bks, gap))
        out = Path(log_dir) / 'cvrp_test_results.csv'
        with out.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['instance', 'customer_count', 'cost', 'best_known', 'gap'])
            writer.writerows(rows)
        return config_id, None
    except Exception as exc:
        return config_id, f'{type(exc).__name__}: {exc}'


def evaluate_instance(heuristic, distance, demands, capacity):
    n = len(demands) - 1
    unvisited = set(range(1, n + 1))
    route, current, load = [0], 0, 0.0
    while unvisited:
        feasible = np.asarray(sorted(x for x in unvisited
                                     if load + demands[x] <= capacity), dtype=int)
        if feasible.size == 0:
            if current != 0:
                route.append(0)
            current, load = 0, 0.0
            continue
        nxt = int(heuristic(current, 0, feasible.copy(), capacity - load,
                            demands.copy(), distance.copy()))
        if nxt == 0:
            if current == 0:
                return None
            route.append(0); current, load = 0, 0.0
            continue
        if nxt not in unvisited or nxt not in feasible:
            return None
        route.append(nxt); unvisited.remove(nxt)
        load += float(demands[nxt]); current = nxt
    if route[-1] != 0:
        route.append(0)
    return float(sum(distance[a, b] for a, b in zip(route, route[1:])))


def main(results_root, test_data_path, workers):
    root = Path(results_root)
    test_data = load_object_npz(test_data_path, 'cvrp_dict')
    jobs = [(i, root / f'{i:02d}', test_data) for i in range(59)]
    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(evaluate_one, jobs)
    for config_id, error in results:
        print(f'{config_id:02d}: {error or "ok"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.results_root, args.test_data, max(1, args.workers))
