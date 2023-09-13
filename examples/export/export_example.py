# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

from examples.models import MODEL_NAME_TO_MODEL
from examples.models.model_factory import EagerModelFactory
from examples.export.utils import export_to_edge, export_to_exec_prog, save_pte_program

# def export_to_pte(model_name, model, method_name, example_inputs):
#     edge = export_to_edge(model, method_name, example_inputs)
#     exec_prog = edge.to_executorch()
#     for node in edge.exported_program.graph.nodes:
#         if str(node) == "aten_index_tensor":
#             print(node.meta)
#     # exir.print_program.pretty_print(exec_prog.program.execution_plan)
#
# FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
# logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-n",
        "--method_name",
        required=False,
        default="forward",
        help=f"[Optional] method name. Default is forward",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    prog = export_to_exec_prog(model, args.method_name, example_inputs)
    save_pte_program(prog.buffer, args.model_name)
