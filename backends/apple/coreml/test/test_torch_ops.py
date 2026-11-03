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

    def _coreml_partitioner(self, *, minimum_deployment_target=ct.target.iOS18):
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=minimum_deployment_target
        )
        return CoreMLPartitioner(compile_specs=compile_specs)

    def _get_test_model(self):
        model = torch.nn.Sequential(
            torch.nn.Embedding(64, 128), torch.nn.Linear(128, 256), torch.nn.ReLU()
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

    def test_int_input_of_float_ops(self):
        """
        Torch computes these ops in float for an int or bool input, and
        coremltools passed the input on, so lowering failed.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                i = x.to(torch.int64)
                j = i + 1
                k = (i > 1).to(torch.int64)
                return torch.stack(
                    [
                        i.sqrt(),
                        i.exp(),
                        i.expm1(),
                        i.sigmoid(),
                        i.sin(),
                        i.cos(),
                        i.tan(),
                        i.sinh(),
                        i.cosh(),
                        i.tanh(),
                        i.atan(),
                        i.erf(),
                        j.rsqrt(),
                        j.reciprocal(),
                        j.log2(),
                        j.log10(),
                        k.asin(),
                        k.acos(),
                        (i - i).atanh(),
                        (i > 1).sqrt(),
                        (i >= 0).log(),
                        (i > 1).log1p(),
                    ]
                )

        # Small integers, so that each op reads the same ints in Core ML.
        model, example_inputs = Model(), (torch.randint(0, 4, (2, 3)).float(),)
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        self._compare_outputs(delegated_program.to_executorch(), model, example_inputs)

    def test_int_input_of_ops_that_keep_it_int(self):
        """
        Torch keeps an int input of floor, ceil, round, relu, relu6 and
        hardtanh int, and coremltools passed it on to Core ML ops that take only
        float, so lowering failed. Int bounds of hardtanh failed on a float16
        input too.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                # -4, 0, 4 and 8, so that relu, relu6 and hardtanh clip on both
                # sides.
                i = x.to(torch.int64) * 4 - 4
                ops = [i.floor(), i.ceil(), i.round(), i.relu()]
                ops += [
                    torch.nn.functional.relu6(i),
                    # Torch drops the fraction of each bound, so this clips at
                    # -1 and 2.
                    torch.nn.functional.hardtanh(i, -1.5, 2.7),
                    # Int bounds on a float16 input.
                    torch.nn.functional.hardtanh(x.half() * 4 - 4, 0, 6).long(),
                ]
                return torch.stack(ops).float()

        model, example_inputs = Model(), (
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]),
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            torch.export.export(model, example_inputs),
            partitioner=[self._coreml_partitioner()],
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        self._compare_outputs(delegated_program.to_executorch(), model, example_inputs)

    def test_hardtanh_of_float_arange(self):
        """
        coremltools gives torch.arange(6, dtype=torch.float32) the dtype int32, so
        its hardtanh failed to lower. Torch clips it in float, at bounds with a
        fraction. A TorchScript node carries no dtype and keeps the coremltools
        converter.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                return torch.nn.functional.hardtanh(a, 0.5, 2.5) + x

        model, example_inputs = Model(), (torch.zeros(2, 3),)
        ep = torch.export.export(model, example_inputs)
        # With no passes, Core ML keeps the clip instead of folding it.
        program = ct.convert(
            ep.run_decompositions({}),
            convert_to="milinternal",
            pass_pipeline=ct.PassPipeline.EMPTY,
        )
        self.assertEqual(
            [(op.alpha.val, op.beta.val) for op in program.find_ops(op_type="clip")],
            [(0.5, 2.5)],
        )
        ct.convert(
            torch.jit.trace(torch.nn.Hardtanh(0.5, 2.5), example_inputs),
            inputs=[ct.TensorType(shape=(2, 3))],
            convert_to="milinternal",
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[self._coreml_partitioner()]
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        self._compare_outputs(delegated_program.to_executorch(), model, example_inputs)

    def test_copy_into_other_dtype(self):
        """
        Torch copies a value into a tensor of another dtype in the dtype of
        the tensor, also for y[:, i] = v, and coremltools kept the dtype of the
        value, so lowering failed.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 2
                y[:, 0] = (x[:, 1] * 4).to(torch.int64)
                y[:, 2] = x[:, 1] > 0.5
                z = x.clone().copy_((x * 4).to(torch.int64))
                # Torch drops the fraction of a float written into an int.
                i = (x * 4).to(torch.int64)
                i[:, 0] = x[:, 1] * 4
                # coremltools makes this arange int32, so the copy has to take
                # the dtype from torch.
                a = torch.arange(6, dtype=torch.float32).reshape(2, 3).copy_(x)
                return y + torch.nn.functional.gelu(z) + i + a.softmax(-1)

        # Values that float16 holds exactly, so that the comparison with 0.5 in
        # Core ML agrees with eager.
        model, example_inputs = Model(), (torch.rand(2, 3).half().float(),)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            torch.export.export(model, example_inputs),
            partitioner=[self._coreml_partitioner()],
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        self._compare_outputs(delegated_program.to_executorch(), model, example_inputs)

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

    def test_dequantize_affine_c8w_embedding_c8w_linear_ios16(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[
                self._coreml_partitioner(minimum_deployment_target=ct.target.iOS16)
            ],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_linear_per_grouped_col(self):
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

    def test_dequantize_codebook_linear_per_grouped_row(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint2, block_size=[16, -1]),
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

    def test_dequantize_codebook_embedding_per_grouped_col(self):
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

    def test_dequantize_codebook_embedding_per_grouped_row(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint3, block_size=[16, -1]),
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

    def test_dequantize_affine_below_ios18_raises_with_hint(self):
        """
        Regression test for https://github.com/pytorch/executorch/issues/13122.

        `quantize_(...)` with blockwise / int4 configurations requires iOS18.
        coremltools raises a ValueError that does not mention how to fix the
        deployment target on the ExecuTorch side; we wrap it to add the
        partitioner-level guidance.
        """
        model = torch.nn.Linear(64, 64)
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model.eval(), (torch.randn(1, 64),), strict=True)
        with self.assertRaises(ValueError) as cm:
            executorch.exir.to_edge_transform_and_lower(
                ep,
                partitioner=[
                    self._coreml_partitioner(minimum_deployment_target=ct.target.iOS17)
                ],
            )
        msg = str(cm.exception)
        self.assertIn("iOS18", msg)
        self.assertIn("CoreMLPartitioner", msg)
        self.assertIn("minimum_deployment_target", msg)


if __name__ == "__main__":
    test_runner = TestTorchOps()
    test_runner.test_dequantize_affine_b4w_embedding()
    test_runner.test_dequantize_affine_b4w_linear()
    test_runner.test_dequantize_affine_c4w_embedding()
    test_runner.test_dequantize_affine_c4w_linear()
    test_runner.test_dequantize_affine_c8w_embedding_b4w_linear()
    test_runner.test_dequantize_affine_c8w_embedding_c8w_linear_ios16()
    test_runner.test_dequantize_codebook_linear_per_grouped_col()
    test_runner.test_dequantize_codebook_linear_per_grouped_row()
    test_runner.test_dequantize_codebook_embedding_per_grouped_col()
    test_runner.test_dequantize_codebook_embedding_per_grouped_row()
    test_runner.test__clone_dim_order_contiguous()
