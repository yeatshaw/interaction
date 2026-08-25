# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import time
import traceback
import random
from threading import Thread, Lock
from typing import Optional, Literal

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


class EoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 info: dict | None = None,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 reflection_parent_info: bool = False,
                 reflection_best_worst: bool = False,
                 reflection_fitness: int | None = None,
                 reflection_avg_fitness: bool = False,
                 reflection_check_guidance: bool = False,
                 lineage_log_path: str = 'eoh_lineage.json',
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._info = dict(info or {})
        self._template_program_str = self._info['template_program']
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode

        self._reflection_parent_info = reflection_parent_info
        self._reflection_best_worst = reflection_best_worst
        self._reflection_fitness = reflection_fitness
        self._reflection_avg_fitness = reflection_avg_fitness
        self._reflection_check_guidance = reflection_check_guidance
        self._reflection_suggestion = None
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        # function to be evolved
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._profiler = profiler
        # The lineage DAG is persisted on disk; population survival remains the
        # sole mechanism for evolution. Only the current population stays in RAM.
        # When the caller leaves the default path, keep the DAG beside the
        # profiler's timestamped run logs (e.g. logs/eoh_tsp/YYYYMMDD_HHMMSS).
        if lineage_log_path == 'eoh_lineage.json' and profiler is not None:
            lineage_log_path = os.path.join(profiler._log_dir, 'eoh_lineage.json')
        self._lineage_log_path = os.path.abspath(lineage_log_path)
        self._lineage_base = os.path.splitext(self._lineage_log_path)[0]
        self._lineage_shard_size = 100
        self._next_lineage_id = 1
        self._lineage_lock = Lock()
        self._initialize_lineage_file()
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)

        # statistics
        self._tot_sample_nums = 0
        self._convergence_history = []
        self._best_score_so_far = float('-inf')

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _reflect(self, children, parents=None):
        """Generate a compact suggestion; reflection text is not registered as a child."""
        prompt = EoHPrompt.get_prompt_reflection(
            children, parents=parents, info=self._info,
            parent_info_flag=self._reflection_parent_info,
            best_worst_flag=self._reflection_best_worst,
            fitness_flag=self._reflection_fitness,
            avg_fitness_flag=self._reflection_avg_fitness,
            check_reflection_flag=self._reflection_check_guidance,
            population=self._population
        )
        print(f'\nReflection Prompt: {prompt}\n')
        try:
            suggestion = self._sampler.llm.draw_sample(prompt)
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return None
        return suggestion.strip() if isinstance(suggestion, str) else None

    def _prepare_reflection(self, refs):
        if not self._population.population:
            return
        lineage_parents = [self.get_lineage_parents(child) for child in refs]
        self._reflection_suggestion = self._reflect(refs, lineage_parents)

    def _register_lineage_node(self, func, parents=None):
        """Add a function to the persistent DAG before population survival."""
        with self._lineage_lock:
            node_id = self._next_lineage_id
            self._next_lineage_id += 1
            parent_ids = []
            for parent in parents or []:
                parent_id = getattr(parent, '_eoh_lineage_id', None)
                if parent_id is not None:
                    parent_ids.append(parent_id)
            func._eoh_lineage_id = node_id
            func._eoh_parent_ids = tuple(parent_ids)
            record = {
                'node_id': node_id,
                'parent_ids': parent_ids,
                'score': func.score,
                'algorithm': getattr(func, 'algorithm', ''),
                'suggestion': getattr(func, '_eoh_generation_suggestion', None),
                'name': func.name,
                'args': func.args,
                'body': func.body,
                'return_type': func.return_type,
                'generation': self._population.generation,
                'operator': getattr(func, 'operator', None),
            }
            self._append_lineage_record(record)

    def _initialize_lineage_file(self):
        path = os.path.dirname(self._lineage_log_path)
        if path:
            os.makedirs(path, exist_ok=True)
        shard_files = self._lineage_shard_files()
        if not shard_files:
            self._write_lineage_shard(1, [])
            return
        # Only the newest shard is needed to continue numbering; parent lookup
        # later opens just the shard containing each requested parent ID.
        filename = max(shard_files, key=lambda item: int(
            os.path.basename(item).rsplit('_', 1)[1].split('~', 1)[0]))
        try:
            with open(filename, encoding='utf-8') as file:
                nodes = json.load(file).get('nodes', [])
            max_id = max((int(node['node_id']) for node in nodes), default=0)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid lineage JSON file: {filename}') from exc
        self._next_lineage_id = max_id + 1

    def _lineage_shard_path(self, node_id):
        start = ((int(node_id) - 1) // self._lineage_shard_size) * self._lineage_shard_size + 1
        end = start + self._lineage_shard_size - 1
        return f'{self._lineage_base}_{start}~{end}.json'

    def _lineage_shard_files(self):
        directory = os.path.dirname(self._lineage_log_path) or '.'
        prefix = os.path.basename(self._lineage_base) + '_'
        return [os.path.join(directory, name) for name in os.listdir(directory)
                if name.startswith(prefix) and name.endswith('.json')]

    def _write_lineage_shard(self, node_id, nodes):
        path = self._lineage_shard_path(node_id)
        directory = os.path.dirname(path) or '.'
        fd, temp_path = tempfile.mkstemp(prefix='.eoh_lineage_', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file:
                json.dump({'nodes': nodes}, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _append_lineage_record(self, record):
        path = self._lineage_shard_path(record['node_id'])
        try:
            with open(path, encoding='utf-8') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'nodes': []}
        data.setdefault('nodes', []).append(record)
        self._write_lineage_shard(record['node_id'], data['nodes'])

    def get_lineage_parents(self, func):
        """Return parent Function objects even after population survival removes them."""
        with self._lineage_lock:
            records = {}
            for parent_id in getattr(func, '_eoh_parent_ids', ()):
                path = self._lineage_shard_path(parent_id)
                try:
                    with open(path, encoding='utf-8') as file:
                        records[int(parent_id)] = next(
                            (node for node in json.load(file).get('nodes', [])
                             if int(node['node_id']) == int(parent_id)), None)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
        parents = []
        for parent_id in getattr(func, '_eoh_parent_ids', ()):
            record = records.get(int(parent_id))
            if record is None:
                continue
            parent = Function(
                name=record['name'], args=record['args'], body=record['body'],
                return_type=record.get('return_type'), score=record.get('score')
            )
            parent.algorithm = record.get('algorithm', '')
            parent.operator = record.get('operator')
            parent._eoh_generation_suggestion = record.get('suggestion')
            parent._eoh_lineage_id = record['node_id']
            parent._eoh_parent_ids = tuple(record.get('parent_ids', []))
            parents.append(parent)
        return parents

    def _sample_evaluate_register(self, prompt, parents=None, operator=None):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        generation_suggestion = self._reflection_suggestion if operator != 'i1' else None
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return False
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return False
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        # register to profiler
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator
        func._eoh_generation_suggestion = generation_suggestion
        self._tot_sample_nums += 1
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
        if score is None:
            return False
        if self._max_sample_nums is None or len(self._convergence_history) < self._max_sample_nums:
            self._best_score_so_far = max(self._best_score_so_far, score)
            self._convergence_history.append(
                (self._tot_sample_nums, self._best_score_so_far)
            )
        self._register_lineage_node(func, parents)

        # register to the population
        self._population.register_function(func)

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_eoh_operator(self):
        while self._continue_loop():
            try:
                # get a new func using e1
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                # 对indivs进行反思得到建议
                self._prepare_reflection(indivs)
                prompt = EoHPrompt.get_prompt_e1(indivs, self._info, self._reflection_suggestion)
                if self._debug_mode:
                    print(f'E1 Prompt: {prompt}')
                self._sample_evaluate_register(prompt, indivs, operator='e1')
                if not self._continue_loop():
                    break
                
                # get a new func using e2
                if self._use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._selection_num)]
                    self._prepare_reflection(indivs)
                    prompt = EoHPrompt.get_prompt_e2(indivs, self._info, self._reflection_suggestion)
                    if self._debug_mode:
                        print(f'E2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, indivs, operator='e2')
                    if not self._continue_loop():
                        break

                # get a new func using m1
                if self._use_m1_operator:
                    indiv = self._population.selection()
                    self._prepare_reflection([indiv])
                    prompt = EoHPrompt.get_prompt_m1(indiv, self._info, self._reflection_suggestion)
                    if self._debug_mode:
                        print(f'M1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, [indiv], operator='m1')
                    if not self._continue_loop():
                        break

                # get a new func using m2
                if self._use_m2_operator:
                    indiv = self._population.selection()
                    self._prepare_reflection([indiv])
                    prompt = EoHPrompt.get_prompt_m2(indiv, self._info, self._reflection_suggestion)
                    if self._debug_mode:
                        print(f'M2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, [indiv], operator='m2')
                    if not self._continue_loop():
                        break
            except KeyboardInterrupt:
                return
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _iteratively_init_population(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = EoHPrompt.get_prompt_i1(self._info)
                self._sample_evaluate_register(prompt, operator='i1')
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    # print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    print(
                        f'Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            # do initialization
            self._multi_threaded_sampling(self._iteratively_init_population)
            self._population.survival()
            # terminate searching if
            if len(self._population) < self._selection_num:
                print(
                    f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return

        # evolutionary search
        # Generations are built as exact pop_size batches; run this coordinator
        # in one thread so survival happens only after each complete batch.
        self._multi_threaded_sampling(self._iteratively_use_eoh_operator())

        # finish
        if self._profiler is not None:
            self._profiler.finish()

        self._sampler.llm.close()
