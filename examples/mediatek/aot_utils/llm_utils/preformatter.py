import json
import os
from typing import Union


class Preformatter(object):
    __slots__ = ("template", "name", "_verbose")

    def __init__(self, template_path: str = "", verbose: bool = False):
        self._verbose = verbose
        self.name = os.path.basename(template_path).rsplit(".", 1)[0]
        if not os.path.exists(template_path):
            raise ValueError(f"Can't read preformatter template json: {template_path}")
        with open(template_path) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_path}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input is not None:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
