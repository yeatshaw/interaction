"""Shared EoH prompt metadata loader for all example tasks."""

from __future__ import annotations

import ast
import importlib
import re
import textwrap
from pathlib import Path


def _signature_from_source(source: str, method_name: str, class_name: str | None = None) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if class_name and not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    args = [ast.unparse(arg) for arg in item.args.args]
                    if args and args[0] == 'self':
                        args = args[1:]
                    return ', '.join(args)
    raise ValueError(f'Method {method_name!r} was not found in the template source.')


def get_info(method_name: str, template_module: str, *, source_file: str | None = None,
             class_name: str | None = None) -> dict:
    """Read a task template module and return the fields required by EoH.

    ``template_module`` is imported by its repository module path, for example
    ``llm4ad.task.optimization.tsp_construct.template``.
    """
    template = importlib.import_module(template_module)
    method_text = getattr(template, 'method_args', '')
    pattern = rf'(?ms)^{re.escape(method_name)}:\s*\n(.*?)(?=^[A-Za-z_]\w*:\s*\n|\Z)'
    match = re.search(pattern, method_text)
    if match is None:
        raise ValueError(f'Unknown method {method_name!r} in {template_module}.')

    if source_file:
        signature = _signature_from_source(
            Path(source_file).read_text(encoding='utf-8'), method_name, class_name
        )
    elif hasattr(template, 'template_program'):
        signature = _signature_from_source(template.template_program, method_name)
    else:
        raise ValueError('Provide source_file when template_program is not defined.')

    template_program = getattr(
        template, 'template_program',
        f'def {method_name}({signature}):\n    pass\n'
    )
    return {
        'method_name': method_name,
        'task_description': template.task_description,
        'class_args': template.class_args.strip(),
        'method_args': textwrap.dedent(match.group(1)).strip(),
        'func_template': template.func_template,
        'method_signature': signature,
        'template_program': template_program,
    }
