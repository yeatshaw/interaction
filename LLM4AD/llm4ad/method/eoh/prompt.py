from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
    @classmethod
    def get_prompt_reflection(cls, children, parents=None, mode: int = 4, info: dict | None = None) -> str:
        """Build the reflection prompt using the task template's path format."""
        task_prompt = info['task_description']
        children = children if isinstance(children, list) else [children]
        parents = parents or [[] for _ in children]
        def code_block(code, label='Code'):
            return f'{label}:\n```python\n{code}\n```'

        def child_section(item, index, include_thought):
            prefix = f'## Algorithm {index} ##\n'
            thought = f'thought: {getattr(item, "algorithm", "")}\n' if include_thought else ''
            return prefix + thought + code_block(str(item))

        if mode in (1, 2):
            sections = '\n'.join(child_section(item, i + 1, mode == 2)
                                  for i, item in enumerate(children))
            evidence = (
                'Here are a few pieces of algorithm code to complete the above tasks.\n'
                if mode == 1 else
                'Here are a few pieces of algorithm code and their corresponding thoughts to complete the above task.\n'
            ) + sections
        else:
            paths = []
            for i, (child, group) in enumerate(zip(children, parents), 1):
                multiple = len(group) > 1
                refs = []
                for j, parent in enumerate(group, 1):
                    suffix = f' {j}' if multiple else ''
                    thought = (f'thought{suffix}: {getattr(parent, "algorithm", "")}\n'
                               if mode == 4 else '')
                    refs.append(f'Code{suffix}:\n```python\n{parent}\n```' if mode == 3
                                else thought + f'Code{suffix}:\n```python\n{parent}\n```')
                new_thought = (f'thought: {getattr(child, "algorithm", "")}\n' if mode == 4 else '')
                paths.append(
                    f'## Evolution path {i} ##\n# Reference Algorithm #\n'
                    + '\n'.join(refs) + '\n# New Algorithm #\n'
                    + new_thought + code_block(str(child))
                )
            evidence = ('Here are the reference algorithm sets from the previous algorithm design and the new algorithms generated from them.\n'
                        + '\n'.join(paths))
        if mode not in (1, 2, 3, 4):
            raise ValueError('reflection_input_mode must be one of 1, 2, 3, or 4.')
        return f'''Task:{task_prompt}
{evidence}
Identify the most useful design insight from the provided algorithm(s). Then, output a specific improvement suggestion within {{}}, without outputting code or extra explanations.
'''
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
{class_args}
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
{class_args}
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
{class_args}
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
{class_args}
{method_args}
This is the format for your reply:
{func_template}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
{cls._requirements()}
Do not give additional explanations.'''
        return prompt_content
