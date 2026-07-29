from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program
from ...base.modify_code import ModifyCode


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = self.__class__.trim_code_from_response(response)

        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            match = re.search(r'(?is)\bthought\s*:\s*(\{.*?\})', response)
            return match.group(1) if match else None
        except:
            return None

    @classmethod
    def trim_code_from_response(cls, response: str) -> str:
        """Extract a function body from either fenced or plain LLM output."""
        match = re.search(r'```(?:python)?\s*(.*?)```', response, flags=re.IGNORECASE | re.DOTALL)
        code = match.group(1) if match else response
        code = re.sub(r'^\s*Code\s*:\s*\n?', '', code, flags=re.IGNORECASE)
        return SampleTrimmer.trim_preface_of_function(code)
