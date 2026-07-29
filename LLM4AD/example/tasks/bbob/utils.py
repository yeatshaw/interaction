"""BBOB prompt metadata selected by the method currently evolved."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from llm4ad.task.optimization.bbob import template


def _method_signature(method_name: str) -> str:
    source = Path(template.__file__).with_name('lshade.py').read_text(encoding='utf-8')
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == 'LSHADE':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    args = [arg.arg for arg in item.args.args]
                    defaults = [ast.unparse(value) for value in item.args.defaults]
                    first_default = len(args) - len(defaults)
                    parts = [name if index < first_default else f'{name}={defaults[index - first_default]}'
                             for index, name in enumerate(args)]
                    if not parts or parts[0] != 'self':
                        raise ValueError(f'Method {method_name!r} must start with self.')
                    return ', '.join(parts[1:])
    raise ValueError(f'Method {method_name!r} was not found in LSHADE.')


def get_info(method_name: str) -> dict:
    """Return the template fields needed by EoH and its prompt for one method."""
    pattern = rf'(?ms)^{re.escape(method_name)}:\s*\n(.*?)(?=^[A-Za-z_]\w*:\s*\n|\Z)'
    match = re.search(pattern, template.method_args)
    if match is None:
        available = re.findall(r'(?m)^([A-Za-z_]\w*):\s*$', template.method_args)
        raise ValueError(f'Unknown method {method_name!r}; available: {", ".join(available)}.')
    signature = _method_signature(method_name)
    return {
        'method_name': method_name,
        'task_description': template.task_description,
        'class_args': template.class_args.strip(),
        'method_args': textwrap.dedent(match.group(1)).strip(),
        'func_template': template.func_template,
        'method_signature': signature,
        'template_program': f'def {method_name}(self, {signature}):\n    pass\n',
    }
