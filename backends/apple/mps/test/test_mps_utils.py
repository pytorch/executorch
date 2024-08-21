#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import unittest

from typing import Any, Tuple

import executorch.exir as exir
import torch
from executorch.backends.apple.mps import MPSBackend
from executorch.backends.apple.mps.partition import MPSPartitioner
from executorch.exir import EdgeCompileConfig, ExirExportedProgram, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge
from executorch.sdk import BundledProgram
from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torch.export import export

# Config for Capturing the weights, will be moved in the future

# TODO(T182928844): Delegate dim order op to backend.
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False, _skip_dim_order=True
)


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


def dump_bundled_program(sample_inputs, expected_output, executorch_program, func_name):
    method_test_suites = [
        MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(inputs=sample_inputs, expected_outputs=expected_output)
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


class TestMPS(unittest.TestCase):
    def assert_outputs_equal(
        self,
        model_output,
        ref_output,
        use_fp16: bool = False,
        atol: float = 1e-03,
        rtol: float = 1e-03,
    ):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the MPS's backend, we relax the detal such that absolute tolerance is 1e-3. and
        relative tolerance is 1e-3.
        """
        # Compare the result from executor and eager mode directly
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            assert len(ref_output) == len(
                model_output
            ), "Length of outputs is not matching!"
            for i in range(len(ref_output)):
                res_output = model_output[i].cpu()
                expected_output = ref_output[i].cpu()
                if use_fp16 and (
                    expected_output.dtype == torch.float16
                    or res_output.dtype == torch.float16
                ):
                    # cast back from fp16 to fp32 (ExecuTorch results are in FP32 by default)
                    expected_output = expected_output.to(torch.float32)
                    res_output = res_output.to(torch.float32)
                if (
                    torch.allclose(res_output, expected_output, atol=atol, rtol=rtol)
                    is False
                ):
                    mean_err = (
                        (res_output - expected_output).abs() / expected_output
                    ).mean()
                    logging.debug(f"mean err = {mean_err}")
                    self.assertLess(mean_err, 0.05)
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            expected_output = ref_output.cpu()
            res_output = model_output[0].cpu()
            if use_fp16 and (
                expected_output.dtype == torch.float16
                or res_output.dtype == torch.float16
            ):
                # cast back from fp16 to fp32 (ExecuTorch results are in FP32 by default)
                expected_output = expected_output.to(torch.float32)
                res_output = res_output.to(torch.float32)
            if (
                torch.allclose(res_output, expected_output, atol=atol, rtol=rtol)
                is False
            ):
                mean_err = (
                    (res_output - expected_output).abs() / expected_output
                ).mean()
                logging.debug(f"mean err = {mean_err}")
                self.assertLess(mean_err, 0.05)

    def lower_module_and_test_output(
        self,
        module: Any,
        sample_inputs: Tuple[torch.Tensor],
        func_name: str,
        use_partitioner: bool = True,
        use_fp16: bool = False,
        bundled_program=True,
        dynamic_shapes=None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
    ) -> ExirExportedProgram:
        """
        Helper testing function that takes a torch.nn.Module and lowers it to MPS with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """
        logging.info("Step 1: EXIR capturing of original module")

        model = module.eval()
        original_inputs = []
        for t in sample_inputs:
            original_inputs.append(t.detach().clone())
        original_inputs = tuple(original_inputs)

        expected_output = model(*sample_inputs)

        model = torch._export.capture_pre_autograd_graph(
            model, sample_inputs, dynamic_shapes=dynamic_shapes
        )

        edge_program = export_to_edge(
            model,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            edge_compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
            ),
        )

        logging.info(
            f"Step 2: Lowering to MPSGraph {'with' if use_partitioner else 'without'} partitioner"
        )
        compile_specs = [CompileSpec("use_fp16", bytes([use_fp16]))]

        if use_partitioner:
            logging.info(f"Edge IR graph:\n{edge_program.exported_program().graph}")
            delegated_program = edge_program
            delegated_program = edge_program.to_backend(
                MPSPartitioner(compile_specs=compile_specs)
            )
            logging.info(
                f"Lowered graph:\n{delegated_program.exported_program().graph}"
            )

            executorch_program = delegated_program.to_executorch(
                config=ExecutorchBackendConfig(
                    extract_delegate_segments=False, extract_constant_segment=False
                )
            )
        else:
            delegated_program = to_backend(
                MPSBackend.__name__, edge_program.exported_program(), compile_specs
            )

            executorch_program = to_edge(
                export(
                    delegated_program,
                    sample_inputs,
                ),
                compile_config=exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
                ),
            ).to_executorch(
                config=ExecutorchBackendConfig(
                    extract_delegate_segments=False, extract_constant_segment=False
                )
            )

        if bundled_program:
            dump_bundled_program(
                sample_inputs, expected_output, executorch_program, func_name
            )
        try:
            from executorch.extension.pybindings.portable_lib import (  # @manual
                _load_for_executorch_from_buffer,
            )

            logging.info("Testing delegated program using pybind")

            # Test the model with executor
            logging.debug("Initializing MPSGraph")
            executorch_module = _load_for_executorch_from_buffer(
                executorch_program.buffer
            )

            model_output = executorch_module.forward(original_inputs)

            logging.info(f"Expected output: {expected_output}")
            logging.info(f"MPS delegate output: {model_output}")
            self.assert_outputs_equal(model_output, expected_output, atol, rtol)
            logging.info("Delegated program matches PyTorch Eager mode result!")

            return delegated_program
        except ImportError:
            logging.info(
                "ExecuTorch MPS delegate was built without pybind support. Exiting..."
            )

    def lower_and_test_with_partitioner(
        self,
        graph_module,
        example_inputs,
        func_name: str,
        use_fp16: bool = False,
        dynamic_shapes=None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
    ):
        logging.info(func_name)
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=True,
            func_name=func_name,
            use_fp16=use_fp16,
            dynamic_shapes=None,
            atol=atol,
            rtol=rtol,
        )

    def lower_and_test_without_partitioner(
        self,
        graph_module,
        example_inputs,
        func_name: str,
        use_fp16: bool = False,
        dynamic_shapes=None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
    ):
        logging.info(func_name)
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=False,
            func_name=func_name,
            use_fp16=use_fp16,
            dynamic_shapes=dynamic_shapes,
            atol=atol,
            rtol=rtol,
        )
