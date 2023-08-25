# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse

import executorch.exir as exir

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from .utils import export_to_edge


def export_to_pte(model_name, model, example_inputs):
    edge = export_to_edge(model, example_inputs)
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
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[args.model_name])

    export_to_pte(args.model_name, model, example_inputs)
