"""Return-value statistics plugins injected outside the DE implementation."""

from __future__ import annotations

import atexit
import json
import os
import types
import weakref
from pathlib import Path

import numpy as np


_RECORDER_STATE: dict[tuple[str, str, int, str], dict] = {}


def instrument_program(program: str, operator_name: str, output_dir: str | Path,
                       candidate_id: str) -> str:
    """Append a same-signature wrapper; the generated operator body is untouched."""
    if not operator_name.isidentifier():
        raise ValueError(f'Invalid operator name: {operator_name!r}.')
    return program + f'''\n\n__statistics_original_{operator_name} = {operator_name}
def {operator_name}(self, *args, **kwargs):
    result = __statistics_original_{operator_name}(self, *args, **kwargs)
    from llm4ad.task.optimization.bbob.operator_statistics_plugin import record_operator_return
    record_operator_return({str(output_dir)!r}, {candidate_id!r}, {operator_name!r}, self, args, result)
    return result
'''


def _state(output_dir: str, candidate_id: str, operator_name: str, owner) -> dict:
    key = (output_dir, candidate_id, os.getpid(), operator_name)
    if key not in _RECORDER_STATE:
        _RECORDER_STATE[key] = {'calls': 0, 'stage': None, 'stages': [{} for _ in range(10)]}
        atexit.register(_flush, key)
        weakref.finalize(owner, _flush, key)
    return _RECORDER_STATE[key]


def _add(stage: dict, name: str, value: float, weight: int) -> None:
    if not np.isfinite(value) or weight <= 0:
        return
    accumulator = stage.setdefault(name, {'sum': 0.0, 'weight': 0})
    accumulator['sum'] += float(value)
    accumulator['weight'] += int(weight)


def _spread(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((x - np.mean(x, axis=0)) ** 2, axis=1))))


def _diversity_ratio(new: np.ndarray, old: np.ndarray) -> float:
    return _spread(new) / max(_spread(old), 1e-12)


def _nearest_old_indices(new: np.ndarray, old: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, for each new individual, its closest individual in the old population."""
    distances = np.linalg.norm(new[:, None, :] - old[None, :, :], axis=2)
    indices = np.argmin(distances, axis=1)
    return indices, distances[np.arange(len(new)), indices]


def _record_mutate(stage: dict, owner, args, result) -> None:
    if len(args) < 2 or not isinstance(result, tuple) or len(result) < 2:
        return
    x, y, x_mu = map(lambda value: np.asarray(value, dtype=float), (args[0], args[1], result[0]))
    step = x_mu - x
    step_norm = np.linalg.norm(step, axis=1)
    spread = _spread(x)
    best_direction = x[np.argmin(y)] - x
    denominator = step_norm * np.linalg.norm(best_direction, axis=1)
    alignment = np.divide(np.sum(step * best_direction, axis=1), denominator,
                          out=np.zeros_like(denominator), where=denominator > 1e-12)
    _add(stage, 'normalized_step_norm', np.sum(step_norm / max(spread, 1e-12)), len(x))
    _add(stage, 'step_norm_std', np.std(step_norm), 1)
    _add(stage, 'best_alignment', np.sum(alignment), len(x))
    _add(stage, 'scaling_factor_mean', np.sum(np.asarray(result[1], dtype=float)), len(x))
    _add(stage, 'offspring_diversity_ratio', _diversity_ratio(x_mu, x), 1)
    f_mu = np.asarray(result[1], dtype=float)
    correlation = 0.0 if np.std(f_mu) < 1e-12 or np.std(step_norm) < 1e-12 else np.corrcoef(f_mu, step_norm)[0, 1]
    _add(stage, 'f_step_correlation', correlation, 1)


def _record_crossover(stage: dict, owner, args, result) -> None:
    if len(args) < 2 or not isinstance(result, tuple) or len(result) < 2:
        return
    x_mu, x, x_cr = map(lambda value: np.asarray(value, dtype=float), (args[0], args[1], result[0]))
    step = x_mu - x
    mutation_energy = max(float(np.sum(step * step)), 1e-12)
    retained_energy = float(np.sum(np.where(np.isclose(x_cr, x_mu), step * step, 0.0)))
    transfer_energy = float(np.sum((x_cr - x) ** 2))
    n = len(x)
    _add(stage, 'energy_retention', retained_energy / mutation_energy, n)
    _add(stage, 'donor_coordinate_retention', np.mean(np.isclose(x_cr, x_mu)), n)
    _add(stage, 'step_transfer_ratio', transfer_energy / mutation_energy, n)
    _add(stage, 'crossover_rate_mean', np.sum(np.asarray(result[1], dtype=float)), n)
    donor_distance = np.linalg.norm(x_mu - x, axis=1)
    trial_distance = np.linalg.norm(x_cr - x, axis=1)
    distance_ratio = np.divide(trial_distance, donor_distance, out=np.zeros_like(trial_distance),
                               where=donor_distance > 1e-12)
    _add(stage, 'donor_parent_distance_ratio', np.sum(distance_ratio), n)
    _add(stage, 'crossover_diversity_ratio', _diversity_ratio(x_cr, x_mu), 1)


def _record_restart(stage: dict, owner, args, result) -> None:
    if len(args) < 3 or not isinstance(result, tuple) or len(result) < 3:
        return
    x, y, x_new, y_new = map(np.asarray, (args[0], args[1], result[0], result[1]))
    a = np.empty((0, x.shape[1])) if args[2] is None else np.asarray(args[2])
    a_new = np.empty((0, x.shape[1])) if result[2] is None else np.asarray(result[2])
    if (x.ndim != 2 or x_new.ndim != 2 or len(x) == 0 or len(x_new) == 0
            or len(y) != len(x) or len(y_new) != len(x_new)):
        return
    nearest_indices, nearest_distances = _nearest_old_indices(x_new, x)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(x))))
    unchanged = nearest_distances <= tolerance
    activated = not np.all(unchanged)
    elite_n = min(len(x), max(1, int(np.ceil(owner.p_min * len(x)))))
    elite_indices = np.argsort(y)[:elite_n]
    elite_retained = np.mean([np.any(nearest_indices[unchanged] == index) for index in elite_indices])
    _add(stage, 'restart_activation_rate', float(activated), 1)
    _add(stage, 'replacement_rate', 1.0 - np.mean(unchanged), 1)
    _add(stage, 'population_size_ratio', len(x_new) / len(x), 1)
    _add(stage, 'population_diversity_ratio', _diversity_ratio(x_new, x), 1)
    _add(stage, 'elite_retention_rate', elite_retained, 1)
    _add(stage, 'restart_improvement', np.min(y) - np.min(y_new), 1)
    _add(stage, 'restart_success_ratio', np.mean(y_new < y[nearest_indices]), 1)
    old_archive_size = len(a) if a.ndim > 1 else 0
    _add(stage, 'archive_size_ratio', len(a_new) / max(old_archive_size, 1), 1)


_EXTRACTORS = {
    'mutate': _record_mutate,
    'crossover': _record_crossover,
    'restart': _record_restart,
}


def install_fixed_operator_recorders(output_dir: str, candidate_id: str,
                                     evolved_operator: str, owner) -> None:
    """Wrap the two fixed operators once for this optimizer instance."""
    marker = '_operator_statistics_fixed_recorders'
    if getattr(owner, marker, False):
        return
    setattr(owner, marker, True)
    for operator_name in _EXTRACTORS:
        if operator_name == evolved_operator:
            continue
        original = getattr(owner, operator_name, None)
        if not callable(original):
            continue

        def wrapped(self, *args, _original=original, _operator_name=operator_name, **kwargs):
            result = _original(*args, **kwargs)
            record_operator_return(output_dir, candidate_id, _operator_name, self, args, result)
            return result

        setattr(owner, operator_name, types.MethodType(wrapped, owner))


def record_operator_return(output_dir: str, candidate_id: str, operator_name: str,
                           owner, args, result) -> None:
    """Called by the injected wrapper after a generated operator returns."""
    extractor = _EXTRACTORS.get(operator_name)
    if extractor is None:
        return
    state = _state(output_dir, candidate_id, operator_name, owner)
    stage_index = min(9, int(10 * owner.n_function_evaluations /
                             max(owner.max_function_evaluations, 1)))
    extractor(state['stages'][stage_index], owner, args, result)
    state['calls'] += 1
    if stage_index != state['stage'] or state['calls'] % 100 == 0:
        state['stage'] = stage_index
        _flush((output_dir, candidate_id, os.getpid(), operator_name))


def _flush(key: tuple[str, str, int, str]) -> None:
    state = _RECORDER_STATE.get(key)
    if state is None:
        return
    output_dir, candidate_id, pid, operator_name = key
    records = []
    for stage_index, values in enumerate(state['stages']):
        if not values:
            continue
        record = {'candidate_id': candidate_id, 'operator_name': operator_name,
                  'process_id': pid, 'stage': stage_index}
        for name, item in values.items():
            if item['weight']:
                record[name] = item['sum'] / item['weight']
                record[f'{name}__weight'] = item['weight']
        records.append(record)
    path = Path(output_dir) / 'raw' / f'{candidate_id}_{operator_name}_{pid}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding='utf-8')
