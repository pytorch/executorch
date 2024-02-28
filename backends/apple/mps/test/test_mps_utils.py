#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import unittest

from typing import Any, Tuple, Union

import executorch.exir as exir
import torch
from executorch.backends.apple.mps.mps_preprocess import MPSBackend
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgram,
    ExirExportedProgram,
    to_edge,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.tracer import Value
from executorch.sdk import BundledProgram
from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

# Config for Capturing the weights, will be moved in the future
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=True)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(_check_ir_validity=False)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs)
    logging.info(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_compile_config=None,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep, compile_config=edge_compile_config
    )

    edge_manager.exported_program().graph.print_tabular()
    logging.info(f"Exported graph:\n{edge_manager.exported_program().graph}")
    return edge_manager


def export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> EdgeProgramManager:
    core_aten_ep = _to_core_aten(model, example_inputs)
    return _core_aten_to_edge(core_aten_ep, edge_compile_config)


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
        use_partitioner: bool = True,
        use_fp16: bool = False,
    ) -> ExirExportedProgram:
        """
        Helper testing function that takes a torch.nn.Module and lowers it to MPS with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """

        logging.info("Step 1: EXIR capturing of original module")

        class WrappedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.one_module = module

            def forward(self, *args):
                return self.one_module(*args)

        model = WrappedModule()
        model = model.eval()
        model = capture_pre_autograd_graph(model, sample_inputs)

        edge_program = export_to_edge(
            model,
            sample_inputs,
            edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        logging.info(
            f"Step 2: Lowering to MPSGraph {'with' if use_partitioner else 'without'} partitioner"
        )
        compile_specs = [CompileSpec("use_fp16", bytes([use_fp16]))]

        if use_partitioner:
            logging.info(f"Edge IR graph:\n{edge_program.exported_program().graph}")
            edge = edge_program.to_backend(MPSPartitioner(compile_specs=compile_specs))
            logging.info(f"Lowered graph:\n{edge.exported_program().graph}")

            executorch_program = edge.to_executorch(
                config=ExecutorchBackendConfig(extract_constant_segment=False)
            )
        else:
            delegated_program = to_backend(
                MPSBackend.__name__, edge_program.exported_program(), compile_specs
            )

            executorch_program = (
                exir.capture(
                    delegated_program,
                    sample_inputs,
                    exir.CaptureConfig(enable_aot=True, _unlift=False),
                )
                .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
                .to_executorch(
                    config=ExecutorchBackendConfig(extract_constant_segment=False)
                )
            )

        exported_program: ExirExportedProgram = exir.capture(
            WrappedModule(), sample_inputs, _CAPTURE_CONFIG
        ).to_edge(_EDGE_COMPILE_CONFIG)

        executorch_program: ExecutorchProgram = exported_program.to_executorch()

        logging.info("Step 3: Generating bundled program")
        logging.info(
            "  -> Number of execution plans: {len(executorch_program.program.execution_plan)}"
        )

        expected_output = module(*sample_inputs)

        method_test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=sample_inputs, expected_outputs=module(*sample_inputs)
                    )
                ],
            )
        ]

        logging.info(f"Expected output: {expected_output}")
        logging.info("  -> Test suites generated successfully")

        bundled_program = BundledProgram(executorch_program, method_test_suites)
        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )

        filename = f"{func_name}.pte"
        logging.info(f"Step 4: Saving bundled program to {filename}")
        with open(filename, "wb") as file:
            file.write(bundled_program_buffer)

    def lower_and_test_with_partitioner(
        self,
        graph_module,
        example_inputs,
        func_name: str,
        use_fp16: bool = False,
    ):
        logging.info(func_name)
        # MPS TODO: partitioner support
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=True,
            func_name=func_name,
            use_fp16=use_fp16,
        )
