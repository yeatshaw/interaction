from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
    @classmethod
    def get_prompt_reflection(cls, children, parents=None, mode: int = 4, info: dict | None = None) -> str:
        """Ask for one concise improvement suggestion for the next algorithm."""
        task_prompt = info['task_description']
        children = children if isinstance(children, list) else [children]
        parents = parents or [[] for _ in children]
        child_code = '\n\n'.join(f'Candidate {i + 1}:\n{item}' for i, item in enumerate(children))
        child_thought = '\n\n'.join(f'Candidate {i + 1}:\n{getattr(item, "algorithm", "")}' for i, item in enumerate(children))
        parent_code = '\n\n'.join(
            f'Candidate {i + 1} parent(s):\n' + '\n\n'.join(str(parent) for parent in group)
            for i, group in enumerate(parents) if group
        )
        parent_thought = '\n\n'.join(
            f'Candidate {i + 1} parent thought(s):\n' + '\n\n'.join(getattr(parent, 'algorithm', '') for parent in group)
            for i, group in enumerate(parents) if group
        )
        if mode == 1:
            evidence = f'Child code:\n{child_code}'
        elif mode == 2:
            evidence = f'Child code:\n{child_code}\n\nChild thought:\n{child_thought}'
        elif mode == 3:
            evidence = f'Parent code(s):\n{parent_code}\n\nChild code:\n{child_code}'
        elif mode == 4:
            evidence = (f'Parent thought(s):\n{parent_thought}\n\nParent code(s):\n{parent_code}\n\n'
                        f'Child thought:\n{child_thought}\n\nChild code:\n{child_code}')
        else:
            raise ValueError('reflection_input_mode must be one of 1, 2, 3, or 4.')
        return f'''{task_prompt}
Extract the key reason in the subject algorithm that should guide the next code change, and turn it into one concrete improvement suggestion.
{evidence}
{cls.requirements()}
Output only the concise improvement suggestion. Do not output the subject-selection reason, analysis, code, thought, or multiple alternatives.'''

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
        required = ('method_name', 'method_args', 'class_args', 'func_template', 'method_signature')
        missing = [key for key in required if key not in info]
        if missing:
            raise ValueError(f'info is missing required fields: {", ".join(missing)}.')
        reply_template = (info['func_template']
                          .replace('<method_name>', info['method_name'])
                          .replace('<method_args>', info['method_signature']))
        return info['method_name'], info['method_args'], reply_template, info['class_args']
    
    @classmethod
    def get_prompt_i1(cls, info: dict | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
This is the information about the variables in this class:
{class_args}
These are the relevant parameters for this method:
{method_args}
This is the format for your reply:
{func_template}
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, indivs: List[Function], info: dict | None = None):
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
This is the information about the variables in this class:
{class_args}
These are the relevant parameters for this method:
{method_args}
This is the format for your reply:
{func_template}
Please help me create a new algorithm that has a totally different form from the given ones. 
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, indivs: List[Function], info: dict | None = None):
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
This is the information about the variables in this class:
{class_args}
These are the relevant parameters for this method:
{method_args}
This is the format for your reply:
{func_template}
Please identify the common backbone idea in the provided methods and help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, indi: Function, info: dict | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'

        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
This is the information about the variables in this class:
{class_args}
These are the relevant parameters for this method:
{method_args}
This is the format for your reply:
{func_template}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
{cls.requirements()}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, indi: Function, info: dict | None = None):
        method_name, method_args, func_template, class_args = cls._template_values(info)
        task_prompt = info['task_description']
        assert hasattr(indi, 'algorithm')
        indiv_prompt = f'{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''{task_prompt} You need to optimize the method '{method_name}' in it.
I have a implementation of this method with its code as follows:
{indiv_prompt}
This is the information about the variables in this class:
{class_args}
These are the relevant parameters for this method:
{method_args}
This is the format for your reply:
{func_template}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
{cls._requirements()}
Do not give additional explanations.'''
        return prompt_content
