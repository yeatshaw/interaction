from __future__ import annotations

import json
import os
import csv
from threading import Lock
from typing import Optional

from .programs_database import ProgramsDatabase
from ...tools.profiler import ProfilerBase
from ...tools.profiler import TensorboardProfiler
from ...tools.profiler import WandBProfiler


class FunSearchProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 initial_num_samples=0,
                 program_db_register_interval: int = 100,
                 checkpoint_interval: int | None = None,
                 convergence_csv_path: str | None = None,
                 convergence_png_path: str | None = None,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """FunSearch Profiler.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        super().__init__(log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._prog_db_order = 0
        if self._log_dir:
            self._prog_db_path = os.path.join(self._log_dir, 'prog_db')
            os.makedirs(self._prog_db_path, exist_ok=True)
        else:
            self._prog_db_path = None
        self._intv = int(program_db_register_interval)
        self._checkpoint_interval = int(
            checkpoint_interval or program_db_register_interval)
        self._convergence_csv_path = convergence_csv_path or (
            os.path.join(self._log_dir, 'convergence.csv')
            if self._log_dir else None)
        self._convergence_png_path = convergence_png_path or (
            os.path.join(self._log_dir, 'convergence.png')
            if self._log_dir else None)
        self._db_lock = Lock()
        self._convergence_lock = Lock()
        self._convergence_records = []
        self._best_so_far = float('-inf')

    def register_function(self, function, program: str = '', *, resume_mode=False):
        sample_order = super().register_function(
            function, program=program, resume_mode=resume_mode)
        if not resume_mode:
            self._record_convergence_sample(function, sample_order)
        return sample_order

    def _record_convergence_sample(self, function, sample_order: int) -> None:
        with self._convergence_lock:
            score = function.score
            self._convergence_records.append({
                'sample_order': int(sample_order),
                'score': None if score is None else float(score),
                'best_so_far': None,
            })
            self._recompute_convergence_locked()

    def _recompute_convergence_locked(self) -> None:
        ordered = sorted(
            self._convergence_records,
            key=lambda record: int(record['sample_order']),
        )
        if not ordered:
            return
        best = float('-inf')
        for record in ordered:
            score = record.get('score')
            if score is not None:
                best = max(best, float(score))
            record['best_so_far'] = None if best == float('-inf') else best
        self._convergence_records = ordered
        self._best_so_far = best

    def register_program_db(
            self,
            program_db: ProgramsDatabase,
            *,
            sample_order: int | None = None,
            tree_reflection=None,
    ):
        """Save ProgramDB to a file.
        [
            [{'score': -300, 'functions': [xxx, xxx, xxx, ...]}, {'score': -200, 'functions': [xxx, xxx, xxx, ...]}, {...}],
            [{...}, {...}],
        ]
        """
        if not self._prog_db_path:
            return None
        sample_order = int(sample_order or self._num_samples)
        try:
            if (sample_order == 0 or
                    sample_order % self._checkpoint_interval != 0):
                return None
            self._db_lock.acquire()
            if getattr(self, '_last_snapshot_sample', None) == sample_order:
                return None
            self._last_snapshot_sample = sample_order
            self._prog_db_order += 1
            isld_list = []
            for island in program_db.islands:
                clus_list = []
                for k, v in island.clusters.items():
                    funcs = [str(f) for f in v.programs]
                    func_dic = {'score': k, 'functions': funcs}
                    clus_list.append(func_dic)
                isld_list.append(clus_list)

            path = os.path.join(self._prog_db_path, f'db_{self._prog_db_order}.json')
            temp_path = path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(isld_list, f)
            os.replace(temp_path, path)

            tree_snapshot = None
            stats_snapshot = None
            experience_snapshot = None
            if tree_reflection is not None:
                tree_snapshot, stats_snapshot, experience_snapshot = (
                    tree_reflection.flush(
                        sample_order=sample_order,
                        force=True,
                    )
                )
            method = None
            if self._parameters is not None and len(self._parameters) >= 3:
                method = self._parameters[2]
            if method is not None and hasattr(method, 'save_checkpoint'):
                return method.save_checkpoint(
                    sample_order=sample_order,
                    database_snapshot=path,
                    tree_snapshot=tree_snapshot,
                    stats_snapshot=stats_snapshot,
                    experience_snapshot=experience_snapshot,
                    checkpoint_order=self._prog_db_order,
                )
            return path
        finally:
            if self._db_lock.locked():
                self._db_lock.release()

    def latest_checkpoint_path(self) -> str | None:
        if not self._log_dir:
            return None
        path = os.path.join(self._log_dir, 'checkpoints', 'checkpoint_latest.json')
        return path if os.path.isfile(path) else None

    def restore_progress(self, checkpoint: dict) -> None:
        with self._convergence_lock:
            self._num_samples = int(checkpoint.get('sample_order', 0))
            best_score = checkpoint.get('best_so_far')
            if best_score is not None:
                self._best_so_far = float(best_score)
                self._cur_best_program_score = float(best_score)
            self._prog_db_order = int(checkpoint.get('checkpoint_order', 0))
            self._last_snapshot_sample = self._num_samples
            self._convergence_records = []
            if self._convergence_csv_path and os.path.isfile(self._convergence_csv_path):
                try:
                    with open(self._convergence_csv_path, 'r', encoding='utf-8', newline='') as f:
                        for row in csv.DictReader(f):
                            self._convergence_records.append({
                                'sample_order': int(row['sample_order']),
                                'score': None if row['score'] == '' else float(row['score']),
                                'best_so_far': None if row['best_so_far'] == ''
                                else float(row['best_so_far']),
                            })
                except (OSError, ValueError, KeyError):
                    self._convergence_records = []
            self._recompute_convergence_locked()

    def _write_convergence_outputs(self) -> None:
        if not self._convergence_csv_path:
            return
        with self._convergence_lock:
            records = list(self._convergence_records)
        records.sort(key=lambda record: int(record['sample_order']))
        directory = os.path.dirname(os.path.abspath(self._convergence_csv_path))
        os.makedirs(directory, exist_ok=True)
        temp_path = self._convergence_csv_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['sample_order', 'score', 'best_so_far'])
            writer.writeheader()
            writer.writerows(records)
        os.replace(temp_path, self._convergence_csv_path)
        if not self._convergence_png_path or not records:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plot_records = [
                row for row in records if row['best_so_far'] is not None]
            if not plot_records:
                return
            x = [row['sample_order'] for row in plot_records]
            y = [row['best_so_far'] for row in plot_records]
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, linewidth=1.5)
            plt.xlabel('sample_order')
            plt.ylabel('best_so_far')
            plt.title('FunSearch Convergence')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self._convergence_png_path, dpi=160)
            plt.close()
        except Exception as exc:
            print(f'WARNING: failed to draw convergence curve: {exc}', flush=True)

    def finish(self):
        self._write_convergence_outputs()


class FunSearchTensorboardProfiler(TensorboardProfiler, FunSearchProfiler):

    def __init__(
            self,
            *,
            log_dir: str | None = None,
            initial_num_samples=0,
            program_db_register_interval: int = 100,
            checkpoint_interval: int | None = None,
            convergence_csv_path: str | None = None,
            convergence_png_path: str | None = None,
            log_style='complex',
            create_random_path=True,
            **kwargs):
        """FunSearch Profiler for Tensorboard.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        FunSearchProfiler.__init__(
            self, log_dir=log_dir,
            program_db_register_interval=program_db_register_interval,
            checkpoint_interval=checkpoint_interval,
            convergence_csv_path=convergence_csv_path,
            convergence_png_path=convergence_png_path,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )
        TensorboardProfiler.__init__(
            self, log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )

    def register_function(self, function, program='', *, resume_mode=False):
        sample_order = TensorboardProfiler.register_function(
            self, function, program=program, resume_mode=resume_mode)
        if not resume_mode:
            self._record_convergence_sample(function, sample_order)
        return sample_order

    def finish(self):
        TensorboardProfiler.finish(self)
        self._write_convergence_outputs()


class FunSearchWandbProfiler(WandBProfiler, FunSearchProfiler):
    def __init__(
            self,
            wandb_project_name: str,
            log_dir: str | None = None,
            *,
            initial_num_samples=0,
            program_db_register_interval: int = 100,
            checkpoint_interval: int | None = None,
            convergence_csv_path: str | None = None,
            convergence_png_path: str | None = None,
            log_style='complex',
            create_random_path=True,
            **kwargs):
        """FunSearch Profiler for Wandb.
        Args:
            wandb_project_name : the name of the wandb project
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        FunSearchProfiler.__init__(
            self,
            log_dir=log_dir,
            program_db_register_interval=program_db_register_interval,
            checkpoint_interval=checkpoint_interval,
            convergence_csv_path=convergence_csv_path,
            convergence_png_path=convergence_png_path,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )
        WandBProfiler.__init__(
            self,
            wandb_project_name=wandb_project_name,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )

    def register_function(self, function, program='', *, resume_mode=False):
        sample_order = WandBProfiler.register_function(
            self, function, program=program, resume_mode=resume_mode)
        if not resume_mode:
            self._record_convergence_sample(function, sample_order)
        return sample_order

    def finish(self):
        WandBProfiler.finish(self)
        self._write_convergence_outputs()
