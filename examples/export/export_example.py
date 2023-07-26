# Example script for exporting simple models to flatbuffer

import argparse

import executorch.exir as exir

import torch
from ..utils import _CAPTURE_CONFIG, _EDGE_COMPILE_CONFIG


class MulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        return input * other

    def get_example_inputs(self):
        return (torch.randn(3, 2), torch.randn(3, 2))


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    def get_example_inputs(self):
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

    def get_example_inputs(self):
        return (torch.ones(1), torch.ones(1))


MODEL_NAME_TO_MODEL = {
    "mul": MulModule,
    "linear": LinearModule,
    "add": AddModule,
    "mv3": None,
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

    if args.model_name == "mv3":
        from executorch.examples.models.mobilenet_v3 import MV3Model

        # Unfortunately lack of consistent interface on example models in this file
        # and how we obtain oss models result in changes like this.
        # we should probably fix this if all the MVP model's export example
        # wiil be added here.
        # For now, to unblock, not planning to land those changes in the current diff
        model = MV3Model.get_model().eval()
        example_inputs = MV3Model.get_example_inputs()
    else:
        model = MODEL_NAME_TO_MODEL[args.model_name]()
        example_inputs = model.get_example_inputs()

    export_to_ff(args.model_name, model, example_inputs)
