# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import coremltools as ct

import executorch.exir

import torch
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.scope import ScopeSource

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    is_fbcode,
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

    def _compare_outputs(
        self, executorch_program, eager_program, example_inputs, atol=1e-2, rtol=1e-2
    ):
        if not IS_VALID_TEST_RUNTIME:
            return
        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        et_outputs = method.execute(example_inputs)
        eager_outputs = eager_program(*example_inputs)
        if isinstance(eager_outputs, torch.Tensor):
            eager_outputs = (eager_outputs,)
        self.assertEqual(len(et_outputs), len(eager_outputs))
        for et_output, eager_output in zip(et_outputs, eager_outputs):
            self.assertTrue(
                torch.allclose(et_output, eager_output, atol=atol, rtol=rtol)
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
                        torch.full((2, 3), 4).sqrt(),
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
                    torch.full((2, 3), 4).floor(),
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

    def test_clamp_min_max_with_bound_of_other_dtype(self):
        """
        Torch computes clamp_min and clamp_max in the dtype it promotes the
        tensor and the bound to, while coremltools required the two to share a
        dtype. So lowering an int bound on a float tensor failed, and so did a
        float bound on an int tensor.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                i = (x * 2).to(torch.int64)
                ops = [x.clamp_min(1), x.clamp_max(1)]
                ops += [i.clamp_min(0.5), i.clamp_max(2.5)]
                ops += [x.clamp_min(i[:1]), i.clamp_max(x[:1] * 2)]
                ops.append(torch.full((2, 3), 0).clamp_min(0.5))
                return torch.stack(ops)

        # Values that float16 holds exactly, so that Core ML drops the same
        # fractions as eager.
        model, example_inputs = Model(), ((torch.rand(2, 3) * 2).half().float(),)
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
                y[:, 1] = torch.full((2,), 7)
                y[:, 2] = x[:, 1] > 0.5
                z = x.clone().copy_((x * 4).to(torch.int64))
                # Torch drops the fraction of a float written into an int.
                i = (x * 4).to(torch.int64)
                i[:, 0] = x[:, 1] * 4
                # coremltools makes this arange int32, so the copy has to take
                # the dtype from torch.
                a = torch.arange(6, dtype=torch.float32).reshape(2, 3).copy_(x)
                w = torch.full((2, 3), 4)
                w[:, 0] = x[:, 0] * 4
                return y + torch.nn.functional.gelu(z) + i + w + a.softmax(-1)

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
        """
        The clones convert to nothing, so the Core ML program returns its own
        input, and one value twice, which Core ML used to return wrong.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 2
                return (
                    torch.ops.dim_order_ops._clone_dim_order(x, dim_order=[0, 1, 2, 3]),
                    y,
                    torch.ops.dim_order_ops._clone_dim_order(y, dim_order=[0, 1, 2, 3]),
                )

        model, example_inputs = Model(), (torch.randn(1, 3, 8, 8),)
        ep = torch.export.export(model, example_inputs)
        mlmodel = CoreMLBackend._convert_to_mlmodel(
            executorch.exir.to_edge(ep).exported_program(),
            CoreMLBackend.generate_compile_specs(),
        )
        description = mlmodel.get_spec().description
        outputs = [output.name for output in description.output]
        self.assertEqual(len(set(outputs)), len(outputs))
        self.assertFalse(set(outputs) & {i.name for i in description.input})
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

    def test_same_shape_expand_output(self):
        """
        coremltools keeps a same-shape expand as a tile when it makes an
        output, and Core ML on the CPU then hands back the value it expands
        wrong, as for the scale_tril that MultivariateNormal returns next to
        its expand. An expand that changes the shape stays a tile.
        """

        class Model(torch.nn.Module):
            def __init__(self, size, expand_first):
                super().__init__()
                self.size = size
                self.expand_first = expand_first

            def forward(self, x):
                y = x * 2
                outputs = (y, y.expand(self.size, -1, -1))
                return outputs[::-1] if self.expand_first else outputs

        # Without the fix, Core ML returns y wrong on the CPU in float32.
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT32,
        )
        example_inputs = (torch.randn(1, 3, 8),)
        for size, expand_first, op_types in (
            (1, False, ["mul", "identity"]),
            (1, True, ["identity", "mul"]),
            (2, False, ["mul", "tile"]),
        ):
            with self.subTest(size=size, expand_first=expand_first):
                model = Model(size, expand_first)
                ep = torch.export.export(model, example_inputs)
                edge_program = executorch.exir.to_edge(ep).exported_program()
                mlmodel = CoreMLBackend._convert_to_mlmodel(edge_program, compile_specs)
                outputs = mlmodel._mil_program.functions["main"].outputs
                self.assertEqual([output.op.op_type for output in outputs], op_types)
                # Debug info attributes each output to the node that makes it.
                self.assertEqual(
                    [
                        output.op.scopes[ScopeSource.EXIR_DEBUG_HANDLE]
                        for output in outputs
                    ],
                    [
                        [node.meta["debug_handle"]]
                        for node in edge_program.graph.output_node().args[0]
                    ],
                )
                delegated_program = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)]
                )
                self._compare_outputs(
                    delegated_program.to_executorch(), model, example_inputs
                )
        # The pass removes the tile it replaces, also when the pass list has
        # no dead_code_elimination to do it.
        mlmodel = CoreMLBackend._convert_to_mlmodel(
            executorch.exir.to_edge(
                torch.export.export(Model(1, False), example_inputs)
            ).exported_program(),
            CoreMLBackend.generate_compile_specs(
                compute_precision=ct.precision.FLOAT32,
                pass_names=["common::const_elimination"],
            ),
        )
        self.assertNotIn(
            "tile",
            [op.op_type for op in mlmodel._mil_program.functions["main"].operations],
        )

    def test_gelu_tanh(self):
        """
        torch.export passes approximate="tanh" to gelu as a keyword, and
        coremltools reads it only from the positional inputs.  If the keyword
        is dropped, the exact gelu runs, which differs by up to 4.7e-4.
        """
        model = torch.nn.GELU(approximate="tanh").eval()
        example_inputs = (torch.linspace(-8, 8, 4096),)
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32
        )
        ep = torch.export.export(model, example_inputs)
        program = ct.convert(ep.run_decompositions({}), convert_to="milinternal")
        self.assertEqual(
            [op.mode.val for op in program.find_ops(op_type="gelu")],
            ["TANH_APPROXIMATION"],
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)]
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                self.assertIn(
                    node.target.__name__, ["executorch_call_delegate", "getitem"]
                )
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs, atol=1e-5, rtol=1e-5)

    def test_fills_keep_their_dtype(self):
        """
        coremltools makes full without a dtype, zeros and new_zeros as float,
        ones_like as float from iOS 16, zeros_like with the dtype of its input,
        and full_like, new_full, fill and scalar_tensor with the type of their
        fill value, instead of the dtype torch gives them. nn.MultiheadAttention
        turns a bool mask into a float one with zeros_like(mask, dtype=...).
        """

        class Fills(torch.nn.Module):
            def forward(self, ids, half):
                ones = torch.ones_like(ids) + ids.new_zeros((2, 3))
                twos = ids.new_full((2, 3), 2.0) + torch.empty_like(ids).fill_(0.0)
                zeros = torch.zeros(2, 3, dtype=torch.long) + torch.full_like(ids, 0.0)
                count = torch.scalar_tensor(1001.0, dtype=torch.long)
                ints = (ids + ones + twos + zeros + torch.full((2, 3), 1)) * count
                flags = torch.where(ids > 0, torch.full((2, 3), True), ids > 1)
                return (
                    ints,
                    flags,
                    half * torch.ones_like(half) + torch.zeros_like(half),
                )

        class Attention(torch.nn.MultiheadAttention):
            def forward(self, x, mask):
                return super().forward(x, x, x, attn_mask=mask, need_weights=False)[0]

        fills_inputs = (torch.arange(6).reshape(2, 3), torch.rand(2, 3).half())
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            torch.export.export(Fills(), fills_inputs),
            partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
        )
        self.assertEqual(
            {
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            },
            {"executorch_call_delegate", "getitem"},
        )
        # The runtime casts each Core ML output to the dtype ExecuTorch expects,
        # so the dtypes are read from the program the delegate converts.
        lowered = delegated_program.exported_program().graph_module.lowered_module_0
        program = ct.convert(
            lowered.original_module,
            convert_to="milinternal",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
        self.assertEqual(
            [output.dtype for output in program.functions["main"].outputs],
            [types.int32, types.bool, types.fp16],
        )
        self._compare_outputs(delegated_program.to_executorch(), Fills(), fills_inputs)
        model = Attention(16, 2, batch_first=True).eval()
        example_inputs = (
            torch.randn(1, 6, 16),
            torch.ones(6, 6, dtype=torch.bool).triu(1),
        )
        attention = torch.export.export(model, example_inputs)
        program = ct.convert(attention.run_decompositions({}), convert_to="milinternal")
        masks = [
            op.outputs[0].dtype
            for op in program.functions["main"].operations
            if op.name.startswith("zeros_like")
        ]
        self.assertTrue(masks and all(types.is_float(dtype) for dtype in masks))
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            attention, partitioner=[CoreMLPartitioner()]
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

    def test_empty_keeps_its_dtype(self):
        """
        coremltools converts empty, new_empty and empty_like as zeros, new_zeros
        and zeros_like, so they need the dtype torch gives them too. With dim
        order ops skipped, as the Core ML export example does, empty reaches
        Core ML as well.
        """

        class Empties(torch.nn.Module):
            def forward(self, ids, x):
                return (
                    torch.empty(2, 3, dtype=torch.long),
                    ids.new_empty((2, 3)),
                    torch.empty_like(x, dtype=torch.long),
                    ids + 1,
                )

        inputs = (torch.arange(6).reshape(2, 3), torch.rand(2, 3))
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            torch.export.export(Empties(), inputs),
            partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
            compile_config=executorch.exir.EdgeCompileConfig(_skip_dim_order=True),
        )
        lowered = delegated_program.exported_program().graph_module.lowered_module_0
        program = ct.convert(
            lowered.original_module,
            convert_to="milinternal",
            compute_precision=ct.precision.FLOAT32,
        )
        self.assertEqual(
            [output.dtype for output in program.functions["main"].outputs],
            [types.int32] * 4,
        )

    def test_fills_of_a_tensor_use_fill_like(self):
        """
        A fill the shape of a tensor becomes a constant as big as that tensor
        in the model when the shape is known at conversion, unless it is a
        fill_like, which needs iOS 16, the target this test uses.
        """

        class Fills(torch.nn.Module):
            def forward(self, x):
                ones = torch.ones_like(x) + torch.zeros_like(x)
                return ones + torch.full_like(x, 2) + x.clone().fill_(3)

        model = Fills().eval()
        x = torch.rand(4, 64)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            torch.export.export(model, (x,)),
            partitioner=[
                self._coreml_partitioner(minimum_deployment_target=ct.target.iOS16)
            ],
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        lowered = delegated_program.exported_program().graph_module.lowered_module_0
        program = ct.convert(
            lowered.original_module,
            convert_to="milinternal",
            minimum_deployment_target=ct.target.iOS16,
        )
        self.assertEqual(len(program.find_ops(op_type="fill_like")), 4)
        self._compare_outputs(delegated_program.to_executorch(), model, (x,))

    @unittest.skipIf(is_fbcode(), "The coremltools in fbcode has no rms_norm.")
    def test_rms_norm_without_eps(self):
        """
        torch.nn.RMSNorm(eps=None) leaves eps out of rms_norm, which
        coremltools requires, and without elementwise_affine the weight too.
        Torch then uses the float32 machine epsilon, which the output for a
        one-hot row of 65536 values depends on.
        """
        x = torch.zeros(1, 65536)
        x[0, 0] = 1
        model = torch.nn.RMSNorm(65536, elementwise_affine=False).eval()
        ep = torch.export.export(model, (x,))
        program = ct.convert(
            ep.run_decompositions({}),
            convert_to="milinternal",
            compute_precision=ct.precision.FLOAT32,
        )
        self.assertEqual(
            [op.y.val for op in program.find_ops(op_type="add")],
            [torch.finfo(torch.float32).eps],
        )
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)]
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                self.assertIn(
                    node.target.__name__, ["executorch_call_delegate", "getitem"]
                )
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, (x,), rtol=1e-3)

    def test_negative_index_next_to_slice(self):
        """
        coremltools writes y[:, i] = v to y[:, i:i + 1] without wrapping a
        negative i, so for i = -1 the slice is empty: lowering failed for a
        constant i, and from iOS 18 Core ML skipped the write for an input i.
        """

        class Write(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.tensor([-1]))

            def forward(self, x, v, i, rows, cols):
                # Two copies, so that a skipped write shows in the output.
                y = x.clone()
                y[:, self.last] = v
                z = x.clone()
                z[:, i] = v * 2
                z[rows, cols] = v[0]
                return y, z

        class Functional(torch.nn.Module):
            def forward(self, x, v, i):
                return torch.ops.aten.index_put(x, [None, i], v)

        model = Write()
        example_inputs = (
            torch.rand(3, 4),
            torch.rand(3, 1),
            torch.tensor([-1]),
            torch.tensor([0, 1]),
            torch.tensor([-1, 2]),
        )
        program = ct.convert(
            torch.export.export(model, example_inputs).run_decompositions({}),
            convert_to="milinternal",
            minimum_deployment_target=ct.target.iOS16,
        )
        # The constant is wrapped while converting, and the write without a
        # slice is left to coremltools, so only the input next to a slice
        # needs a select at run time.
        self.assertEqual(len(program.find_ops(op_type="select")), 1)
        # TorchScript passes the indices as one list, which keeps the
        # coremltools converter.
        ct.convert(
            torch.jit.trace(Functional(), example_inputs[:3]),
            inputs=[
                ct.TensorType(shape=(3, 4)),
                ct.TensorType(shape=(3, 1)),
                ct.TensorType(shape=(1,), dtype=types.int32),
            ],
            convert_to="milinternal",
        )
        width = torch.export.Dim("width", min=2, max=8)
        for dynamic_shapes in (None, ({1: width}, None, None, None, None)):
            with self.subTest(dynamic=dynamic_shapes is not None):
                ep = torch.export.export(
                    model, example_inputs, dynamic_shapes=dynamic_shapes
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
                    ["executorch_call_delegate", "getitem", "getitem"],
                )
                et_program = delegated_program.to_executorch()
                self._compare_outputs(et_program, model, example_inputs)
                if dynamic_shapes is not None:
                    # Another width catches a wrap by the exported width.
                    wide_inputs = (torch.rand(3, 6), *example_inputs[1:])
                    self._compare_outputs(et_program, model, wide_inputs)

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
