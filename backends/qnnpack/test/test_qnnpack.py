# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.exir as exir

import torch
import torch.nn.functional as F

from executorch.backends.qnnpack.partition.qnnpack_partitioner import QnnpackPartitioner

# import the xnnpack backend implementation
from executorch.backends.qnnpack.qnnpack_preprocess import QnnpackBackend
from executorch.exir import CaptureConfig, ExecutorchProgram

from executorch.exir.backend.backend_api import to_backend, validation_disabled

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten

from torch.ao.quantization import QConfig, QConfigMapping  # @manual

from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.observer import (
    default_dynamic_quant_observer,
    default_per_channel_weight_observer,
    default_weight_observer,
)

from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.testing import FileCheck


# Config to use for all calls to `to_executorch()`.
EXECUTORCH_BACKEND_CONFIG = exir.ExecutorchBackendConfig(
    # QNNPACK users should always extract segments during serialization so that
    # the `processed` buffers can be freed after delegate init.
    extract_segments=True,
)

EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(_check_ir_validity=False)


class TestQnnbackends(unittest.TestCase):
    k_dim = 5
    input_dims = (1, 4, k_dim)

    def test_qnnpack_per_channel_dynamic_mm(self):
        linear_mod = torch.nn.Linear(self.k_dim, 4, bias=False).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_mm"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        # Step 3: Lower to QNNPack
        lowered_module = to_backend("QnnpackBackend", captured_mod.exported_program, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        example_inputs = (torch.rand(self.input_dims),)

        composite_model(*example_inputs)
        executorch_program: ExecutorchProgram = (
            exir.capture(composite_model, example_inputs, exir.CaptureConfig())
            .to_edge(EDGE_COMPILE_CONFIG)
            .to_executorch(config=EXECUTORCH_BACKEND_CONFIG)
        )
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))

    def test_qnnpack_per_channel_dynamic_qlinear(self):
        linear_mod = torch.nn.Linear(self.k_dim, 7).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)

        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_channel_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default"
        ).check(
            "aten_view_copy_default"
        ).check(
            "aten_permute_copy_default"
        ).check(
            "aten_addmm_default"
        ).check(
            "aten_view_copy_default"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        # Step 3: Lower to QNNPack
        lowered_module = to_backend("QnnpackBackend", captured_mod.exported_program, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        example_inputs = (torch.rand(self.input_dims),)

        composite_model(*example_inputs)
        executorch_program: ExecutorchProgram = (
            exir.capture(composite_model, example_inputs, exir.CaptureConfig())
            .to_edge(EDGE_COMPILE_CONFIG)
            .to_executorch(config=EXECUTORCH_BACKEND_CONFIG)
        )
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))

    def test_qnnpack_per_tensor_dynamic_mm(self):
        linear_mod = torch.nn.Linear(self.k_dim, 4, bias=False).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)
        captured_mod.exported_program.graph_module.graph.print_tabular()
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_mm"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        # Step 3: Lower to QNNPack
        lowered_module = to_backend("QnnpackBackend", captured_mod.exported_program, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        example_inputs = (torch.rand(self.input_dims),)

        composite_model(*example_inputs)
        executorch_program: ExecutorchProgram = (
            exir.capture(composite_model, example_inputs, capture_config)
            .to_edge(EDGE_COMPILE_CONFIG)
            .to_executorch(config=EXECUTORCH_BACKEND_CONFIG)
        )
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))

    def test_qnnpack_per_tensor_dynamic_qlinear(self):
        linear_mod = torch.nn.Linear(self.k_dim, 4).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)

        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).check(
            "aten_view_copy_default"
        ).check(
            "aten_permute_copy_default"
        ).check(
            "aten_addmm_default"
        ).check(
            "aten_view_copy_default"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        # Step 3: Lower to QNNPack
        lowered_module = to_backend("QnnpackBackend", captured_mod.exported_program, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        example_inputs = (torch.rand(self.input_dims),)

        composite_model(*example_inputs)
        executorch_program: ExecutorchProgram = (
            exir.capture(composite_model, example_inputs, capture_config)
            .to_edge(EDGE_COMPILE_CONFIG)
            .to_executorch(config=EXECUTORCH_BACKEND_CONFIG)
        )
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))

    def test_qnnpack_per_channel_dynamic_mm_with_dynamic_shape(self):
        linear_mod = torch.nn.Linear(self.k_dim, 4, bias=False).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=True)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_mm"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        # Step 3: Lower to QNNPack
        lowered_module = to_backend("QnnpackBackend", captured_mod.exported_program, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        example_inputs = (torch.rand(self.input_dims),)

        composite_model(*example_inputs)
        executorch_program: ExecutorchProgram = (
            exir.capture(composite_model, example_inputs, exir.CaptureConfig())
            .to_edge(EDGE_COMPILE_CONFIG)
            .to_executorch(config=EXECUTORCH_BACKEND_CONFIG)
        )
        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))

    # TODO(someone): Refactor the tests to consolidate reusable code
    def test_qnnpack_per_channel_dynamic_qlinear_via_partitioner(self):
        linear_mod = torch.nn.Linear(self.k_dim, 4).eval()
        example_inputs = (torch.rand(self.input_dims),)
        qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )

        prepared_mod = prepare_fx(
            linear_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(EDGE_COMPILE_CONFIG)

        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_channel_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default"
        ).check(
            "aten_view_copy_default"
        ).check(
            "aten_permute_copy_default"
        ).check(
            "aten_addmm_default"
        ).check(
            "aten_view_copy_default"
        ).run(
            captured_mod.exported_program.graph_module.code
        )

        with validation_disabled():
            lowered_module = captured_mod
            lowered_module.exported_program = to_backend(
                captured_mod.exported_program, QnnpackPartitioner
            )
        FileCheck().check_not(
            "executorch_exir_dialects_edge__ops_aten__to_copy_default"
        ).check_not("executorch_exir_dialects_edge__ops_aten_addmm").run(
            lowered_module.exported_program.graph_module.code
        )

        executorch_program: ExecutorchProgram = lowered_module.to_executorch(
            config=EXECUTORCH_BACKEND_CONFIG
        )

        # TODO(T143084047)
        # class CompositeModule(torch.nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.lowered_module = lowered_module

        #     def forward(self, x):
        #         return self.lowered_module(x)

        # composite_model = CompositeModule()
        # example_inputs = (torch.rand(self.input_dims),)

        # composite_model(*example_inputs)
        # executorch_program: ExecutorchProgram = (
        #     exir.capture(
        #         composite_model, example_inputs, exir.CaptureConfig()
        #     )
        #     .to_edge(EDGE_COMPILE_CONFIG)
        #     .to_executorch()
        # )

        self.assertEqual(
            executorch_program.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )

        # Step 4: Run model and check outputs
        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        inputs_flattened, _ = tree_flatten(example_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = captured_mod(*example_inputs)

        self.assertTrue(torch.allclose(ref_output[0], model_output[0]))
