# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse

import executorch.exir as exir

from ..models import MODEL_NAME_TO_MODEL

from .utils import _CAPTURE_CONFIG, _EDGE_COMPILE_CONFIG


def export_to_ff(model_name, model, example_inputs):
    m = model.eval()
    edge = exir.capture(m, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )
    print("Exported graph:\n", edge.exported_program.graph)

    exec_prog = edge.to_executorch()

    buffer = exec_prog.buffer

    filename = f"{model_name}.pte"
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
