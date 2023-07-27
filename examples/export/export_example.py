# Example script for exporting simple models to flatbuffer

import argparse
from typing import Any, Tuple

import executorch.exir as exir

import torch

from .utils import _CAPTURE_CONFIG, _EDGE_COMPILE_CONFIG


class MulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        return input * other

    @staticmethod
    def get_example_inputs():
        return (torch.randn(3, 2), torch.randn(3, 2))


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    @staticmethod
    def get_example_inputs():
        return (torch.randn(3, 3),)


class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return z

    @staticmethod
    def get_example_inputs():
        return (torch.ones(1), torch.ones(1))


def gen_mobilenet_v3_model_inputs() -> Tuple[torch.nn.Module, Any]:
    # Unfortunately lack of consistent interface on example models in this file
    # and how we obtain oss models result in changes like this.
    # we should probably fix this if all the MVP model's export example
    # wiil be added here.
    # For now, to unblock, not planning to land those changes in the current diff
    from executorch.examples.models.mobilenet_v3 import MV3Model

    return MV3Model.get_model(), MV3Model.get_example_inputs()


def gen_mobilenet_v2_model_inputs() -> Tuple[torch.nn.Module, Any]:
    from executorch.examples.models.mobilenet_v2 import MV2Model

    return MV2Model.get_model(), MV2Model.get_example_inputs()


MODEL_NAME_TO_MODEL = {
    "mul": lambda: (MulModule(), MulModule.get_example_inputs()),
    "linear": lambda: (LinearModule(), LinearModule.get_example_inputs()),
    "add": lambda: (AddModule(), AddModule.get_example_inputs()),
    "mv2": gen_mobilenet_v2_model_inputs,
    "mv3": gen_mobilenet_v3_model_inputs,
}


def export_to_ff(model_name, model, example_inputs):
    m = model
    edge = exir.capture(m, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )
    print("Exported graph:\n", edge.graph)

    exec_prog = edge.to_executorch()

    buffer = exec_prog.buffer

    filename = f"{model_name}.ff"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = MODEL_NAME_TO_MODEL[args.model_name]()

    export_to_ff(args.model_name, model, example_inputs)
