# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig

from ...models import MODEL_NAME_TO_MODEL
from ...models.model_factory import EagerModelFactory
from ..utils import export_to_edge, export_to_exec_prog, save_pte_program


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs, dynamic_shapes = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    if (
        dynamic_shapes is not None
    ):  # capture_pre_autograd_graph does not work with dynamic shapes
        edge_manager = export_to_edge(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            edge_compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        prog = edge_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_constant_segment=False)
        )
    else:
        prog = export_to_exec_prog(model, example_inputs, dynamic_shapes=dynamic_shapes)
    save_pte_program(prog.buffer, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
