# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from random import randint
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from executorch import exir

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_edge_compile_config,
    get_xnnpack_executorch_backend_config,
)
from executorch.backends.xnnpack.utils.utils import capture_graph_for_xnnpack

# import the xnnpack backend implementation
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgram, ExirExportedProgram
from executorch.exir.backend.backend_api import to_backend, validation_disabled

from executorch.exir.passes.spec_prop_pass import SpecPropPass

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten

from torch.ao.quantization import (  # @manual
    default_per_channel_symmetric_qnnpack_qconfig,
    PlaceholderObserver,
    QConfig,
    QConfigMapping,
)

from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)

from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    #    default_weight_observer,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.qconfig_mapping import (
    _get_default_qconfig_mapping_with_default_qconfig,
    _get_symmetric_qnnpack_qconfig_mapping,
)

from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export_for_training

from torch.testing import FileCheck


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


def save_bundled_program(
    representative_inputs, executorch_program, ref_output, output_path
):
    niter = 1

    print("generating bundled program inputs / outputs")

    method_test_cases: List[MethodTestCase] = []
    for _ in range(niter):
        method_test_cases.append(
            MethodTestCase(
                inputs=representative_inputs,
                expected_outputs=ref_output,
            )
        )

    method_test_suites = [
        MethodTestSuite(method_name="forward", method_test_cases=method_test_cases)
    ]

    print("creating bundled program...")
    bundled_program = BundledProgram(executorch_program, method_test_suites)

    print("serializing bundled program...")
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )
    output_path_with_postfix = f"{output_path}_bundled.pte"
    print(f"saving bundled program to {output_path}...")

    with open(output_path_with_postfix, "wb") as file:
        file.write(bundled_program_buffer)


class TestXNNPACK(unittest.TestCase):
    def assert_outputs_equal(self, model_output, ref_output):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the XNNPACK's backend, we relax the detal such that absolute tolerance is 1e-3. and
        relative tolerance is 1e-3.
        """

        # Compare the result from executor and eager mode direclty
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            self.assertTrue(len(ref_output) == len(model_output))
            for i in range(len(ref_output)):
                self.assertTrue(
                    torch.allclose(
                        model_output[i], ref_output[i], atol=1e-03, rtol=1e-03
                    )
                )
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            self.assertTrue(
                torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03)
            )

    def lower_module_and_test_output(
        self,
        module: Any,
        sample_inputs: Tuple[torch.Tensor],
        use_partitioner: bool = False,
        quantized: bool = False,
        quantized_dynamic: bool = False,
        # TODO: remove this after we migrate to use long term flow
        quantizer_api_test: bool = False,
        dump_bundled_program: bool = False,  # for debugging, dump the generated bundled program file
    ) -> ExirExportedProgram:
        """
        Helper testing function that takes a torch.nn.Module and lowers it to XNNPACK with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """

        if quantizer_api_test:
            assert isinstance(module, ExirExportedProgram)
            edge_program = module
        else:

            class WrappedModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.one_module = module

                def forward(self, *args):
                    return self.one_module(*args)

            edge_program = capture_graph_for_xnnpack(WrappedModule(), sample_inputs)

        partitioner = None
        if quantized:
            if quantized_dynamic:
                partitioner = XnnpackDynamicallyQuantizedPartitioner()
            else:
                partitioner = XnnpackPartitioner()
        else:
            partitioner = XnnpackPartitioner()

        if use_partitioner:
            with validation_disabled():
                delegated_program = edge_program
                delegated_program.exported_program = to_backend(
                    edge_program.exported_program, partitioner
                )

            executorch_program: ExecutorchProgram = delegated_program.to_executorch(
                get_xnnpack_executorch_backend_config([SpecPropPass()]),
            )
        else:
            delegated_program = to_backend(
                "XnnpackBackend", edge_program.exported_program, []
            )

            exported_program: ExirExportedProgram = capture_graph_for_xnnpack(
                delegated_program, sample_inputs
            )
            executorch_program: ExecutorchProgram = exported_program.to_executorch(
                get_xnnpack_executorch_backend_config(),
            )

        # print("Graph Module with delegate:")
        # delegated_module.print_readable()

        # Assert the backend name is xnnpack
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            XnnpackBackend.__name__,
        )

        ref_output = delegated_program(*sample_inputs)
        if dump_bundled_program:
            save_bundled_program(
                representative_inputs=sample_inputs,
                executorch_program=executorch_program,
                ref_output=ref_output,
                output_path=f"/tmp/xnnpack_test_{randint(1, 99999)}",
            )

        # Test the model with executor
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        inputs_flattened, _ = tree_flatten(sample_inputs)

        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))

        self.assert_outputs_equal(model_output, ref_output)

        return delegated_program

    def lower_and_test_with_partitioner(
        self,
        graph_module,
        example_inputs,
        quantized: bool = False,
        quantized_dynamic: bool = False,
    ):
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=True,
            quantized=quantized,
            quantized_dynamic=quantized_dynamic,
        )
        self.lower_module_and_test_output(
            graph_module,
            example_inputs,
            use_partitioner=False,
            quantized=quantized,
            quantized_dynamic=quantized_dynamic,
        )

    def quantize_and_test_model(
        self,
        module,
        example_inputs,
        per_channel_quant=False,
    ):
        if per_channel_quant:
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = _get_default_qconfig_mapping_with_default_qconfig(
                False, "qnnpack", qconfig
            )
        else:
            qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
        module.eval()
        prepared = prepare_fx(
            module,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted = _convert_to_reference_decomposed_fx(
            prepared,
            backend_config=get_executorch_backend_config(),
        )

        # Let's assert quant flow did something (not care what, but anything) for this module.
        # This is to ensure we are not just passing through an unquantized model.
        FileCheck().check("torch.ops.quantized_decomposed").run(converted.code)

        self.lower_module_and_test_output(
            module=converted,
            sample_inputs=example_inputs,
            use_partitioner=True,
            quantized=True,
        )

    # TODO: replace quantize_and_test_model with this after
    # QNNPACKQuantizer is more mature
    def quantize_and_test_model_with_quantizer(
        self,
        module,
        example_inputs,
    ):
        module.eval()
        # program capture

        m = export_for_training(
            module,
            example_inputs,
        ).module()

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config()
        quantizer.set_global(quantization_config)
        prepared = prepare_pt2e(m, quantizer)
        converted = convert_pt2e(prepared)

        captured_program = exir.capture(
            converted,
            example_inputs,
            config=exir.CaptureConfig(enable_aot=True, _unlift=True),
        )

        edge_program = captured_program.to_edge(
            get_xnnpack_edge_compile_config()
        ).transform(*get_transform_passes())
        delegated_module = self.lower_module_and_test_output(
            module=edge_program,
            sample_inputs=example_inputs,
            use_partitioner=True,
            quantized=True,
            quantizer_api_test=True,
        )
        supported_ops = {
            "torch.ops.aten.addmm.default",
            "torch.ops.aten.convolution.default",
            "torch.ops.aten.relu.default",
            "torch.ops.aten.add.Tensor",
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor",
        }
        for op in supported_ops:
            FileCheck().check_count(op, 0, exactly=True).run(
                delegated_module.exported_program.graph_module.code
            )

    def _test_xnnpack_dqlinear(
        self,
        weight_qconfig,
        use_bias: bool,
        dump_bundled_program: bool = False,
    ):
        assert weight_qconfig in [
            weight_observer_range_neg_127_to_127,
            per_channel_weight_observer_range_neg_127_to_127,
        ]
        in_size = 2
        input_size = 4
        output_size = 5
        linear = torch.nn.Linear(input_size, output_size, bias=use_bias)
        linear.weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        if use_bias:
            linear.bias = torch.nn.Parameter(torch.rand(output_size))
        example_inputs = (torch.rand(3, in_size, input_size, dtype=torch.float),)
        act_affine_quant_obs = PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=act_affine_quant_obs,
                weight=weight_qconfig,
            ),
        )

        prepared_linear = prepare_fx(
            linear,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_linear = _convert_to_reference_decomposed_fx(
            prepared_linear,
        )

        captured_dqlinear = capture_graph_for_xnnpack(converted_linear, example_inputs)

        captured_dqlinear.exported_program.graph_module.graph.print_tabular()

        lowered_module = to_backend(
            "XnnpackBackend", captured_dqlinear.exported_program, []
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        composite_model(*example_inputs)

        exported_program: ExirExportedProgram = capture_graph_for_xnnpack(
            composite_model, example_inputs
        )
        executorch_program: ExecutorchProgram = exported_program.to_executorch(
            get_xnnpack_executorch_backend_config(),
        )

        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            XnnpackBackend.__name__,
        )

        ref_output = captured_dqlinear(*example_inputs)
        ref_output = composite_model(*example_inputs)
        print("ref_output:", ref_output)

        if dump_bundled_program:
            mm_str = "addmm" if use_bias else "mm"
            filename = f"/tmp/dqlinear_{mm_str}"
            if weight_qconfig == weight_observer_range_neg_127_to_127:
                filename = f"{filename}_per_tensor"
            else:
                filename = f"{filename}_per_channel"

            save_bundled_program(
                representative_inputs=example_inputs,
                executorch_program=executorch_program,
                ref_output=ref_output,
                output_path=filename,
            )

        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        inputs_flattened, _ = tree_flatten(example_inputs)

        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)
        print("ref_output (composite):", ref_output)

        print("Model_output:", model_output[0])

        # Compare the result from executor and eager mode directly
        self.assertTrue(
            torch.allclose(model_output[0], ref_output, atol=4e-03, rtol=1e-03)
        )

    def _get_dqlinear_graph_module(self, weight_qconfig, linear, example_inputs):
        act_affine_quant_obs = PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=act_affine_quant_obs,
                weight=weight_qconfig,
            ),
        )

        prepared_linear = prepare_fx(
            linear,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_dqlinear: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_linear, backend_config=get_executorch_backend_config()
        )

        return converted_dqlinear

    def _test_xnnpack_dqlinear_with_partitioner(self, weight_qconfig, use_bias=True):
        in_size = 1
        input_size = 4
        output_size = 5
        linear = torch.nn.Linear(input_size, output_size, bias=use_bias)
        linear.weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        if use_bias:
            linear.bias = torch.nn.Parameter(torch.rand(output_size))
        example_inputs = (torch.rand(in_size, input_size, dtype=torch.float),)
        converted_dqlinear = self._get_dqlinear_graph_module(
            weight_qconfig, linear, example_inputs
        )

        self.lower_and_test_with_partitioner(
            graph_module=converted_dqlinear,
            example_inputs=example_inputs,
            quantized=True,
            quantized_dynamic=True,
        )

    def _test_xnnpack_custom_dqlinear_with_partitioner_only(
        self, LinearModule, example_inputs
    ):
        linear = LinearModule()
        weight_qconfig = per_channel_weight_observer_range_neg_127_to_127
        converted_dqlinear = self._get_dqlinear_graph_module(
            weight_qconfig, linear, example_inputs
        )

        # Only run test with partitioner
        self.lower_module_and_test_output(
            module=converted_dqlinear,
            sample_inputs=example_inputs,
            use_partitioner=True,
            quantized=True,
            quantized_dynamic=True,
        )
