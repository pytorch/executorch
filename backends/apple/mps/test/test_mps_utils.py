#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import unittest

from typing import Any, Tuple

import executorch.exir as exir

import torch
import torch._export as export
from examples.portable.utils import export_to_edge
from executorch.backends.apple.mps.mps_preprocess import MPSBackend
from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgram, ExirExportedProgram
from executorch.exir.backend.backend_api import to_backend, validation_disabled

# Config for Capturing the weights, will be moved in the future
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=True)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(_check_ir_validity=False)


class ansi_colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class OpSequencesAddConv2d(torch.nn.Module):
    """
    Module which include sequences of Memory Format sensitive ops. forward runs
    [num_sequences] sequences of [ops_per_sequences] ops. Each sequence is
    followed by an add to separate the sequences
    """

    def __init__(self, num_sequences, ops_per_sequence):
        super().__init__()
        self.num_ops = num_sequences * ops_per_sequence
        self.num_sequences = num_sequences
        self.op_sequence = [[] for _ in range(num_sequences)]
        for seq in range(num_sequences):
            for _ in range(ops_per_sequence):
                self.op_sequence[seq].append(
                    torch.nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(3, 3),
                        padding=1,
                        bias=False,
                    )
                )

    def forward(self, x):
        for seq in self.op_sequence:
            for op in seq:
                x = op(x)
            x = x + x
        return x + x


def randomize_bn(num_features: int, dimensionality: int = 2) -> torch.nn.Module:
    if dimensionality == 1:
        bn = torch.nn.BatchNorm1d(num_features)
        input_size = (1, num_features, 5)
    elif dimensionality == 2:
        bn = torch.nn.BatchNorm2d(num_features)
        input_size = (1, num_features, 5, 5)
    else:
        raise AssertionError(
            f"Only dimensionality 1 or 2 supported in randomize_bn, got {dimensionality}"
        )

    bn.weight = torch.nn.Parameter(torch.randn(num_features))
    bn.bias = torch.nn.Parameter(torch.randn(num_features))

    for _ in range(5):
        bn(torch.randn(size=input_size))

    return bn


class TestMPS(unittest.TestCase):
    def lower_module_and_test_output(
        self,
        module: Any,
        sample_inputs: Tuple[torch.Tensor],
        func_name: str,
        use_partitioner: bool = False,
    ) -> ExirExportedProgram:
        """
        Helper testing function that takes a torch.nn.Module and lowers it to XNNPACK with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """

        logging.info("Step 1: EXIR capturing of original module...")

        class WrappedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.one_module = module

            def forward(self, *args):
                return self.one_module(*args)

        m = export.capture_pre_autograd_graph(WrappedModule(), sample_inputs)

        edge_program = export_to_edge(
            m, sample_inputs, edge_compile_config=_EDGE_COMPILE_CONFIG
        )

        logging.info("Step 2: Lowering to MPSGraph...")
        if use_partitioner:
            with validation_disabled():
                None
        else:
            delegated_program = to_backend(
                "MPSBackend", edge_program.exported_program(), []
            )

        logging.info("Step 3: Capturing executorch program with lowered module...")

        class WrappedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mps_module = delegated_program

            def forward(self, *args):
                return self.mps_module(*args)

        exported_program: ExirExportedProgram = exir.capture(
            WrappedModule(), sample_inputs, _CAPTURE_CONFIG
        ).to_edge(_EDGE_COMPILE_CONFIG)

        executorch_program: ExecutorchProgram = exported_program.to_executorch()

        # Assert the backend name is mps
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            MPSBackend.__name__,
        )

        logging.info("Step 4: Generating bundled program...")
        logging.info(
            "  -> Number of execution plans: {len(executorch_program.program.execution_plan)}"
        )
        bundled_inputs = [
            [sample_inputs]
            for _ in range(len(executorch_program.program.execution_plan))
        ]
        logging.info("  -> Bundled inputs generated successfully")

        output = module(*sample_inputs)
        expected_outputs = [
            [[output]] for _ in range(len(executorch_program.program.execution_plan))
        ]
        logging.info("  -> Bundled outputs generated successfully")

        method_names = ["forward"]
        bundled_config = BundledConfig(method_names, bundled_inputs, expected_outputs)
        bundled_program = create_bundled_program(
            executorch_program.program, bundled_config
        )
        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )

        filename = f"{func_name}.pte"
        logging.info(f"Step 5: Saving bundled program to {filename}...")
        with open(filename, "wb") as file:
            file.write(bundled_program_buffer)

    def lower_and_test_with_partitioner(
        self,
        graph_module,
        example_inputs,
        func_name: str,
    ):
        logging.info(func_name)
        # MPS TODO: partitioner support
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=False,
            func_name=func_name,
        )
