# Example script for exporting simple models to flatbuffer

import argparse

import executorch.exir as exir

import torch


class MulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        return input * other

    def get_random_inputs(self):
        return (torch.randn(3, 2), torch.randn(3, 2))


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    def get_random_inputs(self):
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

    def get_random_inputs(self):
        return (torch.ones(1), torch.ones(1))


MODEL_NAME_TO_MODEL = {
    "mul": MulModule,
    "linear": LinearModule,
    "add": AddModule,
}


def export_to_ff(model_name, model):
    m = model()
    edge = exir.capture(
        m, m.get_random_inputs(), exir.CaptureConfig(enable_dynamic_shape=True)
    ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=True))
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
    model = MODEL_NAME_TO_MODEL[args.model_name]

    export_to_ff(args.model_name, model)
