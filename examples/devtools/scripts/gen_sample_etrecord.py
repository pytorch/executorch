# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Generate fixture files
import argparse
import copy
from typing import Any

import torch
from executorch.devtools import generate_etrecord
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    ExportedProgram,
    to_edge,
)
from torch.export import export

from ...models import MODEL_NAME_TO_MODEL
from ...models.model_factory import EagerModelFactory


DEFAULT_OUTPUT_PATH = "/tmp/etrecord.bin"


def gen_etrecord(model: torch.nn.Module, inputs: Any, output_path=None):
    f = model
    aten_dialect: ExportedProgram = export(
        f,
        inputs,
    )
    edge_program: EdgeProgramManager = to_edge(
        aten_dialect, compile_config=EdgeCompileConfig(_check_ir_validity=True)
    )
    edge_program_copy = copy.deepcopy(edge_program)
    et_program: ExecutorchProgramManager = edge_program_copy.to_executorch()
    generate_etrecord(
        (DEFAULT_OUTPUT_PATH if not output_path else output_path),
        edge_dialect_program=edge_program,
        executorch_program=et_program,
        export_modules={
            "aten_dialect_output": aten_dialect,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        help=f"Provide an output path to save the generated etrecord. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    gen_etrecord(model, example_inputs, args.output_path)


if __name__ == "__main__":
    main()  # pragma: no cover
