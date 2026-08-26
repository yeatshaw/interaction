from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
    @classmethod
    def get_prompt_reflection(cls, children, parents=None, info=None,
                              parent_info_flag=False, best_worst_flag=False,
                              fitness_flag=0, avg_fitness_flag=False,
                              check_reflection_flag=False, population=None,
                              use_long_term_reflection=True) -> str:
        """Build a reflection prompt for the input ablation study.

        Defaults are the baseline: selected child code only. ``fitness_flag``
        is 0 (code), 1 (code + thought), or 2 (code + thought + score).
        """
        if info is None:
            raise ValueError('info is required for reflection prompts.')
        children = children if isinstance(children, list) else [children]
        parent_groups = parents or [[] for _ in children]
        if len(parent_groups) != len(children):
            raise ValueError('parents must contain one parent group per child.')
        if fitness_flag not in (0, 1, 2):
            raise ValueError('fitness_flag must be 0, 1, or 2.')

        # Parent-aware reflection uses the evolution-path template only when
        # at least one selected algorithm has a recorded parent. Algorithms
        # without parents are omitted from that mixed input, as required by
        # the ablation design.
        has_parent = parent_info_flag and any(parent_groups)
        if has_parent:
            selected = [(child, group) for child, group in zip(children, parent_groups)
                        if group]
            children = [child for child, _ in selected]
            parent_groups = [group for _, group in selected]
        else:
            parent_groups = [[] for _ in children]

        def display_score(func):
            score = getattr(func, 'score', None)
            if score is None:
                return None
            # EoH maximizes negative cost internally. Reflection exposes the
            # equivalent positive cost so that lower values are better.
            return -score

        def node_label(func, fallback):
            node_id = getattr(func, '_eoh_lineage_id', None)
            return f'Algorithm {node_id}' if node_id is not None else fallback

        def block(func, include_score=True, include_guidance=False):
            text = ''
            if fitness_flag >= 1:
                thought = str(getattr(func, 'algorithm', '') or '').strip()
                # Records created by older runs may contain a formatting label
                # in the stored thought. Keep the reflection field clean.
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

        sections = []
        if has_parent:
            sections.append(
                '===== parent vs. child =====\n'
                'Here are the reference algorithm sets from the previous algorithm design '
                'and the new algorithms generated from them to complete the above task.'
            )
        else:
            sections.append(
                '===== reference =====\n'
                'Here are a few pieces of algorithm to complete the above task.'
            )
        for i, child in enumerate(children, 1):
            child_label = f'Algorithm {i}'
            if has_parent:
                group = parent_groups[i - 1]
                path_label = ('## Evolution path ##' if len(children) == 1
                              else f'## Evolution path {i} ##')
                section = [path_label]
                for j, parent in enumerate(group, 1):
                    parent_label = ('# Reference Algorithm #' if len(group) == 1
                                    else f'# Reference Algorithm {j} #')
                    section.append(parent_label)
                    section.append(block(parent, include_guidance=True))
                section.append('# Generated Algorithm #')
                section.append(block(child, include_guidance=True))
            else:
                section = [f'## {child_label} ##',
                           block(child, include_guidance=True)]
            sections.append('\n'.join(section))

        population_items = list(getattr(population, 'population', population or []))
        if best_worst_flag and population_items:
            ranked = sorted(population_items, key=display_score)
            sections.append('===== best vs. worst =====\n'
                            'Here are the best and worst algorithms in the current population\n'
                            '## Best Algorithm ##\n' +
                            block(ranked[0], include_guidance=True) + '\n' +
                            '## Worst Algorithm ##\n' +
                            block(ranked[-1], include_guidance=True))
        if avg_fitness_flag and population_items:
            scores = [f.score for f in population_items if f.score is not None]
            if scores:
                average_score = -sum(scores) / len(scores)
                sections.append(
                    f'Population average score: {average_score}'
                )

        return (f"Task Description: {info['task_description']}\n"
                + '\n'.join(sections) + '\n'
                "1.Lower scores indicate better algorithms.\n"
                "2.Identify the most useful design insight and output one specific "
                "improvement suggestion inside {}. Do not output code or extra explanations.\n")

    @classmethod
    def get_prompt_design(cls, info, indivs, suggestion):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} method and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
Here is a suggestion to guide the design of a new algorithm:
{suggestion}
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_experience(cls, info, references):
        """Build the long-term reflection prompt from reference histories."""
        paths = []
        for ref in references:
            suggestion = getattr(ref, '_eoh_generation_suggestion', None)
            experience = getattr(ref, '_eoh_experience', None)
            if suggestion or experience:
                paths.append((suggestion, experience))
        if not paths:
            return None
        sections = []
        for index, (suggestion, experience) in enumerate(paths, 1):
            label = ('## Evolution path ##' if len(paths) == 1
                     else f'## Evolution path {index} ##')
            lines = [label]
            if experience:
                lines.append(
                    'The experience gained from evolving algorithms along this path '
                    f'is {experience}.'
                )
            if suggestion:
                lines.append(
                    'The most recent design algorithm guidance for this evolution path '
                    f'is {suggestion}.'
                )
            sections.append('\n'.join(lines))
        return (f"Task Description: {info['task_description']}\n"
                + '\n'.join(sections) + '\n\n'
                'Based on the above, sum up some experiences that align with evolutionary trends.')

    @staticmethod
    def append_reflection(prompt: str, suggestion: str | None, requirements: str) -> str:
        if not suggestion:
            return prompt
        return f'''{prompt}
Use the following reflection as guidance for this generation:
{suggestion}
{requirements}'''

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
    
    @staticmethod
    def requirements() -> str:
        return """Requirements:
    1.Use a sentence inside the {} after 'thought:' to describe your algorithm.
    2.Implement the method after 'Code:'.
    3.If you use any python library or module such as numpy, import it inside the method body before first use. Do not assume imports outside this method exist.
    4.Do not rely on hidden global variables, hidden class fields, or methods that are not visible in the prompt outside the standard python library."""

    # Keep compatibility with the existing m2 prompt text.
    _requirements = requirements

    @staticmethod
    def _template_values(info: dict):
        """Build every prompt field from task metadata."""
        required = ('method_name', 'func_template', 'method_signature')
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
        return info['method_name'], f'{method_section}', reply_template, f'{class_section}'
    
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
    def get_prompt_e1(cls, indivs: List[Function], info: dict | None = None, suggestion: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} method and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
These are some suggestions after reflecting on the given algorithms:
{suggestion}
Please refer to the given suggestions and help me create a new algorithm that has a totally different form from the given ones. 
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, indivs: List[Function], info: dict | None = None,
                      suggestion: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} method and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have {len(indivs)} implementations of this method with their codes as follows:
{indivs_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
These are some suggestions after reflecting on the given algorithms:
{suggestion}
Please refer to the given suggestions and identify the common backbone idea in the provided methods and help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, indi: Function, info: dict | None = None,
                      suggestion: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'

        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
These are some suggestions after reflecting on the given algorithms:
{suggestion}
Please refer to the given suggestions and assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, indi: Function, info: dict | None = None,
                      suggestion: str | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
{class_args}
{method_args}
This is the format for your reply:
{func_template}
These are some suggestions after reflecting on the given algorithms:
{suggestion}
Please refer to the given suggestions and identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
{cls._requirements()}
Do not give additional explanations.'''
        return prompt_content
