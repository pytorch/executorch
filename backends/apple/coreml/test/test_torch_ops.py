# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import coremltools as ct

import executorch.exir

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.backend.utils import format_delegated_graph

from torchao.prototype.quantization.codebook_coreml import CodebookWeightOnlyConfig
from torchao.quantization import IntxWeightOnlyConfig, PerAxis, PerGroup, quantize_

if IS_VALID_TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestTorchOps(unittest.TestCase):
    edge_compile_config = executorch.exir.EdgeCompileConfig()

    def _coreml_partitioner(self):
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        return CoreMLPartitioner(compile_specs=compile_specs)

    def _get_test_model(self):
        model = torch.nn.Sequential(
            torch.nn.Embedding(64, 128), torch.nn.Linear(128, 128), torch.nn.ReLU()
        )
        example_inputs = (torch.LongTensor([0]),)
        return model, example_inputs

    def _compare_outputs(self, executorch_program, eager_program, example_inputs):
        if not IS_VALID_TEST_RUNTIME:
            return
        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        et_outputs = method.execute(example_inputs)[0]
        eager_outputs = eager_program(*example_inputs)
        self.assertTrue(
            torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02)
        )

    def test_dequantize_affine_b4w_embedding(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_b4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c4w_embedding(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0))
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c8w_embedding_b4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint2, block_size=[-1, 16]),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_embedding(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint3, block_size=[-1, 16]),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test__clone_dim_order_contiguous(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.dim_order_ops._clone_dim_order(
                    x, dim_order=[0, 1, 2, 3]
                )

        model, example_inputs = Model(), (torch.randn(1, 3, 8, 8),)
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)


if __name__ == "__main__":
    test_runner = TestTorchOps()
    test_runner.test_dequantize_affine_b4w_embedding()
    test_runner.test_dequantize_affine_b4w_linear()
    test_runner.test_dequantize_affine_c4w_embedding()
    test_runner.test_dequantize_affine_c4w_linear()
    test_runner.test_dequantize_affine_c8w_embedding_b4w_linear()
    test_runner.test_dequantize_codebook_linear()
    test_runner.test_dequantize_codebook_embedding()
    test_runner.test__clone_dim_order_contiguous()
