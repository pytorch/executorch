# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

import torch
from executorch.backends.arm.arm_backend import generate_ethosu_compile_spec

from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig

from ..portable.utils import export_to_edge, save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# TODO: When we have a more reliable quantization flow through to
#       Vela, and use the models in their original form with a
#       quantization step in our example. This will take the models
#       from examples/models/ and quantize then export to delegate.


# Two simple models
class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

    example_input = (torch.ones(5, dtype=torch.int32),)
    can_delegate = True


class AddModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

    example_input = (
        torch.ones(5, dtype=torch.int32),
        torch.ones(5, dtype=torch.int32),
    )
    can_delegate = True


class AddModule3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return (x + y, x + x)

    example_input = (
        torch.ones(5, dtype=torch.int32),
        torch.ones(5, dtype=torch.int32),
    )
    can_delegate = True


class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        z = self.softmax(x)
        return z

    example_input = (torch.ones(2, 2),)
    can_delegate = False


models = {
    "add": AddModule,
    "add2": AddModule2,
    "add3": AddModule3,
    "softmax": SoftmaxModule,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(models.keys())}",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing ArmBackend delegated model",
    )

    args = parser.parse_args()

    if args.model_name not in models.keys():
        raise RuntimeError(f"Model {args.model_name} is not a valid name.")

    if (
        args.model_name in models.keys()
        and args.delegate is True
        and models[args.model_name].can_delegate is False
    ):
        raise RuntimeError(f"Model {args.model_name} cannot be delegated.")

    model = models[args.model_name]()
    example_inputs = models[args.model_name].example_input

    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    edge = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )
    logging.info(f"Exported graph:\n{edge.exported_program().graph}")

    if args.delegate is True:
        edge = edge.to_backend(
            ArmPartitioner(generate_ethosu_compile_spec("ethos-u55-128"))
        )
        logging.info(f"Lowered graph:\n{edge.exported_program().graph}")

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_constant_segment=False)
    )

    model_name = f"{args.model_name}" + (
        "_arm_delegate" if args.delegate is True else ""
    )
    save_pte_program(exec_prog.buffer, model_name)
