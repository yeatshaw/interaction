from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class MAPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''


    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        prompt_content = f'''{task_prompt}
1. First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n'
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value.
1. First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n'
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.
1. Firstly, list the common ideas in the No.{1} algorithm that may give good performances.
2. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_s1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No. {i + 1} algorithm's description and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n"
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.
1. Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm.
2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content


class MAEOHPrompt(MAPrompt):
    """MCTS-AHD prompt builder using the EoH reply/input style.

    This class intentionally lives in the MCTS-AHD prompt module instead of
    importing EoHPrompt.  The original MAPrompt methods stay unchanged for
    ordinary MCTS-AHD runs.
    """

    _COMPARISON_INSTRUCTION = (
        'Compare the provided algorithms, identify their important design differences, '
        'and determine which differences are associated with better or worse performance.'
    )
    _COMPARISON_TASKS = {
        'reference': 'Compare the provided reference algorithms and identify useful design differences.',
        'parent_child': 'Compare parent and child algorithms and identify changes associated with performance differences.',
        'elite_worst': 'Compare the population elite and worst algorithms and identify the structural reasons for their quality gap.',
        'elite_average': 'Compare the population elite with an average-quality algorithm.',
        'worst_average': 'Compare the population worst algorithm with an average-quality algorithm.',
        'children_same_parents': 'Compare multiple children whose complete parent sets are identical.',
        'children_shared_parent': 'Compare multiple children that share at least one parent.',
    }
    _ATTRIBUTION_TASKS = {
        'good': 'Attribute good performance or performance improvements to concrete algorithm design decisions supported by the input.',
        'bad': 'Attribute poor performance or performance degradation to concrete algorithm design decisions supported by the input.',
        'both': 'Attribute both beneficial and harmful outcomes to concrete algorithm design decisions supported by the input.',
        'difference': 'Attribute observed performance differences to specific differences between the algorithms or conditions.',
        'same': 'Explain why the algorithms obtain similar performance despite their design differences.',
    }
    _SUMMARY_TASKS = {
        'guidance': 'Summarize the provided information into concrete guidance for the next algorithm generation.',
        'experience': 'Summarize the provided cases into transferable heuristic experience.',
        'conditions': 'Extract design principles and state the conditions under which each principle applies.',
    }

    @staticmethod
    def requirements() -> str:
        return """Requirements:
    1.Use a sentence inside the {} after 'thought:' to describe your algorithm.
    2.Implement the method after 'Code:'.
    3.If you use any python library or module such as numpy, import it inside the method body before first use. Do not assume imports outside this method exist.
    4.Do not rely on hidden global variables, hidden class fields, or methods that are not visible in the prompt outside the standard python library."""

    _requirements = requirements

    @staticmethod
    def _template_values(info: dict):
        required = ('method_name', 'func_template', 'method_signature', 'task_description')
        missing = [key for key in required if key not in info]
        if missing:
            raise ValueError(f'info is missing required fields: {", ".join(missing)}.')
        reply_template = (info['func_template']
                          .replace('<method_name>', info['method_name'])
                          .replace('<method_args>', info['method_signature']))
        class_args = info.get('class_args', '').strip()
        method_args = info.get('method_args', '').strip()
        class_section = (f'This is the information about the variables in this class:\n{class_args}'
                         if class_args else '')
        method_section = (f'These are the relevant parameters for this method:\n{method_args}'
                          if method_args else '')
        return info['method_name'], method_section, reply_template, class_section

    @classmethod
    def _processing_block(cls, behavior_type='comparison', behavior_task=None):
        tables = {'comparison': cls._COMPARISON_TASKS,
                  'attribution': cls._ATTRIBUTION_TASKS,
                  'summarization': cls._SUMMARY_TASKS}
        if behavior_type not in tables:
            raise ValueError(f'Unknown behavior_type {behavior_type!r}.')
        if behavior_type == 'comparison':
            return f'Comparison: {cls._COMPARISON_INSTRUCTION}'
        table = tables[behavior_type]
        task = behavior_task or next(iter(table))
        if task not in table:
            raise ValueError(f'Unknown {behavior_type} task {task!r}; choose from {", ".join(table)}.')
        labels = {
            'comparison': 'Comparison',
            'attribution': 'Attribution',
            'summarization': 'Summary',
        }
        return f'{labels[behavior_type]}: {table[task]}'

    @staticmethod
    def _reflection_output_block() -> str:
        return ('Based on the above thoughts, output one specific improvement suggestion inside {}. '
                'Do not output code, or extra explanations.')

    @staticmethod
    def _reflection_input_block(parent_info_flag, best_worst_flag,
                                fitness_flag, avg_fitness_flag):
        blocks = ['selected candidate code']
        if parent_info_flag:
            blocks.append('parent candidates when available')
        if best_worst_flag:
            blocks.append('best and worst population candidates')
        if fitness_flag >= 1:
            blocks.append('candidate thought')
        if fitness_flag >= 2:
            blocks.append('candidate score')
        if avg_fitness_flag:
            blocks.append('population mean score')
        return '\n'.join(f'- {block}' for block in blocks)

    @staticmethod
    def _parent_identity(parent):
        return getattr(parent, '_eoh_lineage_id', None) or id(parent)

    @staticmethod
    def _build_parent_child_input(children, parent_groups, algorithm_block):
        selected = [(child, group) for child, group in zip(children, parent_groups)
                    if group]
        sections = [
            '===== parents vs. children =====\n'
            'Here are the reference algorithm sets from the previous algorithm design '
            'and the new algorithms generated from them to complete the above task.'
        ]
        for i, (child, group) in enumerate(selected, 1):
            path_label = '## Evolution path ##' if len(selected) == 1 else f'## Evolution path {i} ##'
            section = [path_label]
            for j, parent in enumerate(group, 1):
                parent_label = '# Reference Algorithm #' if len(group) == 1 else f'# Reference Algorithm {j} #'
                section.extend((parent_label, algorithm_block(parent, include_guidance=False)))
            section.extend(('# Generated Algorithm #', algorithm_block(child, include_guidance=True)))
            sections.append('\n'.join(section))
        return sections

    @staticmethod
    def _build_population_pair_input(title, description, first_label, first,
                                     second_label, second, algorithm_block):
        return [
            f'===== {title} =====\n{description}\n'
            f'## {first_label} ##\n{algorithm_block(first, include_guidance=False)}\n'
            f'## {second_label} ##\n{algorithm_block(second, include_guidance=False)}'
        ]

    @staticmethod
    def _population_representatives(population_items, score_key):
        ranked = sorted(population_items, key=score_key)
        mean_score = sum(score_key(item) for item in ranked) / len(ranked)
        average = min(ranked, key=lambda item: abs(score_key(item) - mean_score))
        return ranked[0], ranked[-1], average

    @classmethod
    def _build_elite_worst_input(cls, population_items, algorithm_block, score_key):
        ranked = sorted(population_items, key=score_key)
        return cls._build_population_pair_input(
            'elite vs. worst',
            'Here are the elite and worst algorithms in the current population.',
            'Elite Algorithm', ranked[0], 'Worst Algorithm', ranked[-1], algorithm_block)

    @classmethod
    def _build_elite_average_input(cls, population_items, algorithm_block, score_key):
        elite, _, average = cls._population_representatives(population_items, score_key)
        return cls._build_population_pair_input(
            'elite vs. population-average individual',
            'The population-average individual is the real individual whose score is closest to the population mean.',
            'Elite Algorithm', elite, 'Population-average Algorithm', average, algorithm_block)

    @classmethod
    def _build_worst_average_input(cls, population_items, algorithm_block, score_key):
        _, worst, average = cls._population_representatives(population_items, score_key)
        return cls._build_population_pair_input(
            'worst vs. population-average individual',
            'The population-average individual is the real individual whose score is closest to the population mean.',
            'Worst Algorithm', worst, 'Population-average Algorithm', average, algorithm_block)

    @classmethod
    def _build_sibling_input(cls, children, parent_groups, algorithm_block,
                             allow_identical=True, allow_shared=True):
        if len(children) < 2 or any(not group for group in parent_groups):
            return None, None
        parent_sets = [{cls._parent_identity(parent) for parent in group}
                       for group in parent_groups]
        identical = all(group == parent_sets[0] for group in parent_sets[1:])
        shared_parent_ids = set.intersection(*parent_sets)
        shared = bool(shared_parent_ids)
        if identical and allow_identical:
            title = 'children with identical parents'
            description = 'These children have exactly the same complete parent set.'
            task = 'children_same_parents'
        elif shared and allow_shared:
            title = 'children with a shared parent'
            description = 'These children share at least one parent.'
            task = 'children_shared_parent'
        else:
            return None, None
        sections = [f'===== {title} =====\n{description}']
        if identical:
            for index, parent in enumerate(parent_groups[0], 1):
                sections.append(
                    f'## Parent Algorithm {index} ##\n'
                    f'{algorithm_block(parent, include_guidance=False)}')
            for index, child in enumerate(children, 1):
                sections.append(
                    f'## Child Algorithm {index} ##\n'
                    f'{algorithm_block(child, include_guidance=True)}')
        else:
            shared_parents = []
            seen = set()
            for parent in parent_groups[0]:
                parent_id = cls._parent_identity(parent)
                if parent_id in shared_parent_ids and parent_id not in seen:
                    shared_parents.append(parent)
                    seen.add(parent_id)
            for index, parent in enumerate(shared_parents, 1):
                sections.append(
                    f'## Shared Parent Algorithm {index} ##\n'
                    f'{algorithm_block(parent, include_guidance=False)}')
            for child_index, (child, group) in enumerate(zip(children, parent_groups), 1):
                sections.append(
                    f'## Child Algorithm {child_index} ##\n'
                    f'{algorithm_block(child, include_guidance=True)}')
                non_shared = [parent for parent in group
                              if cls._parent_identity(parent) not in shared_parent_ids]
                if non_shared:
                    subsection = [f'## Non-shared Parents of Child Algorithm {child_index} ##']
                    for parent_index, parent in enumerate(non_shared, 1):
                        subsection.extend((
                            f'# Non-shared Parent {parent_index} #',
                            algorithm_block(parent, include_guidance=False),
                        ))
                    sections.append('\n'.join(subsection))
        return sections, task

    @classmethod
    def get_prompt_reflection(cls, refs=None, parents=None, info=None,
                              parent_info_flag=False, best_worst_flag=False,
                              fitness_flag=0, avg_fitness_flag=False,
                              check_reflection_flag=False, population=None,
                              use_long_term_reflection=False,
                              behavior_type=None, behavior_task=None,
                              identical_parent_children_flag=False,
                              shared_parent_children_flag=False,
                              population_comparison=None,
                              comparison_flag=None,
                              attribution_flag=None,
                              summarization_flag=None,
                              attribution_task='good',
                              summarization_task='guidance') -> str:
        if info is None:
            raise ValueError('info is required for reflection prompts.')
        children = refs if isinstance(refs, list) else [refs]
        parent_groups = parents or [[] for _ in children]
        if len(parent_groups) != len(children):
            raise ValueError('parents must contain one parent group per reference algorithm.')
        if fitness_flag not in (0, 1, 2):
            raise ValueError('fitness_flag must be 0, 1, or 2.')
        if all(flag is None for flag in (comparison_flag, attribution_flag, summarization_flag)):
            behavior_type = behavior_type or 'comparison'
            comparison_enabled = behavior_type == 'comparison'
            attribution_enabled = behavior_type == 'attribution'
            summarization_enabled = behavior_type == 'summarization'
            if attribution_enabled and behavior_task:
                attribution_task = behavior_task
            if summarization_enabled and behavior_task:
                summarization_task = behavior_task
        else:
            comparison_enabled = bool(comparison_flag)
            attribution_enabled = bool(attribution_flag)
            summarization_enabled = bool(summarization_flag)
        if not any((comparison_enabled, attribution_enabled, summarization_enabled)):
            raise ValueError('At least one reflection behavior must be enabled.')

        behavior_task = behavior_task or 'parent_child'
        identical_parent_children_flag |= behavior_task == 'children_same_parents'
        shared_parent_children_flag |= behavior_task == 'children_shared_parent'
        sibling_comparison_enabled = (
            comparison_enabled and
            (identical_parent_children_flag or shared_parent_children_flag)
        )
        has_parent = (comparison_enabled and
                      (parent_info_flag or sibling_comparison_enabled) and
                      any(parent_groups))

        def display_score(func):
            score = getattr(func, 'score', None)
            return None if score is None else -score

        def block(func, include_score=True, include_guidance=False):
            text = ''
            if fitness_flag >= 1:
                thought = str(getattr(func, 'algorithm', '') or '').strip()
                if thought == 'Code:':
                    thought = ''
                text += f'thought: {thought}\n'
            if fitness_flag >= 2 and include_score:
                text += f'score: {display_score(func)}\n'
            if include_guidance and check_reflection_flag:
                guide = getattr(func, '_eoh_generation_suggestion', None)
                if guide:
                    text += f'This algorithm is generated after being guided by {guide}\n'
                experience = (getattr(func, '_eoh_experience', None)
                              if use_long_term_reflection else None)
                if experience:
                    text += f'This path evolved into the algorithm, and the summarized experience is {experience}.\n'
            if hasattr(func, 'to_code_without_docstring'):
                source = func.to_code_without_docstring().rstrip()
            else:
                source = str(func).rstrip()
            text += f'Code:\n```python\n{source}\n```'
            return text

        population_items = list(getattr(population, 'population', population or []))
        population_tasks = {
            'elite_worst': cls._build_elite_worst_input,
            'elite_average': cls._build_elite_average_input,
            'worst_average': cls._build_worst_average_input,
        }
        sections = []
        sibling_sections = None
        if comparison_enabled and sibling_comparison_enabled:
            sibling_sections, detected_task = cls._build_sibling_input(
                children, parent_groups, block,
                allow_identical=identical_parent_children_flag,
                allow_shared=shared_parent_children_flag)
            if sibling_sections:
                sections.extend(sibling_sections)
                behavior_task = detected_task
        if not sibling_sections and has_parent:
            sections.extend(cls._build_parent_child_input(children, parent_groups, block))
            behavior_task = 'parent_child'
        elif not sibling_sections:
            sections.append(
                '===== reference =====\n'
                'Here are a few pieces of algorithm to complete the above task.'
            )
            for i, child in enumerate(children, 1):
                sections.append('\n'.join((f'## Algorithm {i} ##',
                                           block(child, include_guidance=True))))
            behavior_task = 'reference'

        if comparison_enabled and population_comparison is None and best_worst_flag:
            population_comparison = 'elite_worst'
        if comparison_enabled and population_comparison is not None:
            if population_comparison not in population_tasks:
                raise ValueError('population_comparison must be elite_worst, elite_average, or worst_average.')
            if not population_items:
                raise ValueError(f'{population_comparison} comparison requires a population.')
            sections.extend(population_tasks[population_comparison](
                population_items, block, display_score))
        if comparison_enabled and avg_fitness_flag and population_items:
            scores = [func.score for func in population_items if func.score is not None]
            if scores:
                sections.append(f'Population average score: {-sum(scores) / len(scores)}')

        processing_blocks = []
        if comparison_enabled:
            processing_blocks.append(cls._processing_block('comparison', behavior_task))
        if attribution_enabled:
            processing_blocks.append(cls._processing_block('attribution', attribution_task))
        if summarization_enabled:
            processing_blocks.append(cls._processing_block('summarization', summarization_task))

        return (f"===== Task Description =====\n{info['task_description']}\n"
                + cls._reflection_input_block(
                    parent_info_flag if comparison_enabled else False,
                    best_worst_flag if comparison_enabled else False,
                    fitness_flag,
                    avg_fitness_flag if comparison_enabled else False) + '\n'
                + '\n'.join(sections) + '\n'
                "===== Processing Instruction =====\n"
                + '\n'.join(processing_blocks) + '\n'
                "Lower scores indicate better algorithms.\n"
                "===== Output Format =====\n"
                + cls._reflection_output_block() + '\n')

    @classmethod
    def _format_ref_algorithms(cls, indivs: List[Function]):
        text = ''
        for i, indi in enumerate(indivs, 1):
            assert hasattr(indi, 'algorithm')
            indi.docstring = ''
            text += f'No. {i} method and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n'
        return text

    @staticmethod
    def _suggestion_section(suggestion: str | None,
                            title='These are some suggestions after reflecting on the given algorithms:'):
        if not suggestion:
            return ''
        return f'{title}\n{suggestion}\n'

    @classmethod
    def get_prompt_i1(cls, info: dict | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, indivs: List[Function], info: dict | None = None,
                      suggestion: str | None = None,
                      suggestion_title: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        indivs_prompt = cls._format_ref_algorithms(indivs)
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls._suggestion_section(suggestion, suggestion_title or 'These are some suggestions after reflecting on the given algorithms:')}
Please refer to the given suggestions and create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, indivs: List[Function], info: dict | None = None,
                      suggestion: str | None = None,
                      suggestion_title: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        indivs_prompt = cls._format_ref_algorithms(indivs)
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls._suggestion_section(suggestion, suggestion_title or 'These are some suggestions after reflecting on the given algorithms:')}
Please refer to the given suggestions and create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, indi: Function, info: dict | None = None,
                      suggestion: str | None = None,
                      suggestion_title: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls._suggestion_section(suggestion, suggestion_title or 'These are some suggestions after reflecting on the given algorithms:')}
Please refer to the given suggestions and create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, indi: Function, info: dict | None = None,
                      suggestion: str | None = None,
                      suggestion_title: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls._suggestion_section(suggestion, suggestion_title or 'These are some suggestions after reflecting on the given algorithms:')}
Please refer to the given suggestions and identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_s1(cls, indivs: List[Function], info: dict | None = None,
                      suggestion: str | None = None,
                      suggestion_title: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        indivs_prompt = cls._format_ref_algorithms(indivs)
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
{cls._suggestion_section(suggestion, suggestion_title or 'These are some suggestions after reflecting on the given algorithms:')}
Please refer to the given suggestions and help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content
