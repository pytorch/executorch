#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

# Example script for exporting simple models to flatbuffer

import argparse
import logging

import torch

import torch._export as export
from executorch import exir
from executorch.backends.apple.mps.mps_preprocess import MPSBackend

from executorch.exir.backend.backend_api import to_backend
from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.core import create_bundled_program
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from ....models import MODEL_NAME_TO_MODEL
from ....models.model_factory import EagerModelFactory

from ....portable.utils import save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-b",
        "--bundled",
        action="store_true",
        required=False,
        default=False,
        help="Flag for bundling inputs and outputs in the final flatbuffer program",
    )
    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}.")

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = export.capture_pre_autograd_graph(model, example_inputs)

    edge = exir.capture(
        model, example_inputs, exir.CaptureConfig(enable_aot=True, _unlift=True)
    ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")

    lowered_module = to_backend(MPSBackend.__name__, edge.exported_program, [])

    logging.info(f"Lowered graph:\n{edge.exported_program.graph}")

    class WrappedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mps_module = lowered_module

        def forward(self, *input_args):
            return self.mps_module(*input_args)

    executorch_program = (
        exir.capture(
            WrappedModule(),
            example_inputs,
            exir.CaptureConfig(enable_aot=True, _unlift=True),
        )
        .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
        .to_executorch()
    )

    model_name = f"{args.model_name}_mps"

    if args.bundled:
        method_test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=example_inputs, expected_outputs=[model(*example_inputs)]
                    )
                ],
            )
        ]

        bundled_program = create_bundled_program(executorch_program, method_test_suites)
        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )
        model_name = f"{model_name}_bundled"
        program_buffer = bundled_program_buffer
    else:
        program_buffer = executorch_program.buffer

    save_pte_program(program_buffer, model_name)
