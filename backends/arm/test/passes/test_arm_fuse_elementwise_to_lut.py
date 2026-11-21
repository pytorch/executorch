# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _SigmoidTanh(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(torch.sigmoid(x))


def _quantized_region(dtype=torch.int8, arithmetic=False, scalar_input=False):
    class InputWithConstant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("constant", torch.tensor([[7]], dtype=dtype))

        def forward(self, x):
            return x + self.constant

    info = torch.iinfo(dtype)
    domain = (
        torch.arange(info.min, info.max + 1, dtype=torch.int32).to(dtype).reshape(1, -1)
    )
    if scalar_input:
        domain = domain[:, :1]
    program = to_edge(export(InputWithConstant(), (domain,))).exported_program()
    graph = program.graph
    constant, source = [node for node in graph.nodes if node.op == "placeholder"]
    sink = next(node for node in graph.nodes if node.op == "call_function")
    input_qparams = QuantArgs(
        0.05 if dtype == torch.int8 else 0.0001, 3, info.min, info.max, dtype
    )
    hidden_qparams = QuantArgs(
        0.01 if dtype == torch.int8 else 0.00005, -10, info.min, info.max, dtype
    )
    output_qparams = QuantArgs(hidden_qparams.scale, -7, info.min, info.max, dtype)
    with graph.inserting_before(sink):
        first = graph.call_function(exir_ops.edge.aten.sigmoid.default, (source,))
        first.meta = sink.meta.copy()
        first.meta["input_qparams"] = {0: input_qparams}
        first.meta["output_qparams"] = {0: hidden_qparams}
        previous = first
        if arithmetic:
            constant_qparams = QuantArgs(0.05, 2, info.min, info.max, dtype)
            for target in (
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.sub.Tensor,
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.div.Tensor,
            ):
                node = graph.call_function(target, (previous, constant))
                node.meta = sink.meta.copy()
                node.meta["input_qparams"] = {0: hidden_qparams, 1: constant_qparams}
                node.meta["output_qparams"] = {0: hidden_qparams}
                previous = node
    sink.target = exir_ops.edge.aten.tanh.default
    sink.args = (previous,)
    sink.meta["input_qparams"] = {0: hidden_qparams}
    sink.meta["output_qparams"] = {0: output_qparams}
    graph.eliminate_dead_code()
    graph.lint()
    program.graph_module.recompile()
    return program, source, first, sink, domain


class FuseElementwiseToLUTTests(unittest.TestCase):
    def test_quantized_arithmetic_dequantizes_only_constant_operands(self):
        from executorch.backends.arm.tosa.specification import (
            TosaLoweringContext,
            TosaSpecification,
        )

        program, source, first, sink, domain = _quantized_region(arithmetic=True)
        input_qparams = first.meta["input_qparams"][0]
        hidden_qparams = first.meta["output_qparams"][0]
        output_qparams = sink.meta["output_qparams"][0]
        value = torch.sigmoid(input_qparams.dequantize_value(domain))
        previous = hidden_qparams.quantize_value(value)
        for op in (torch.add, torch.sub, torch.mul, torch.div):
            value = op(value, torch.tensor(0.25))
            previous = hidden_qparams.quantize_value(
                op(hidden_qparams.dequantize_value(previous), torch.tensor(0.25))
            )
        reference = torch.tanh(value)
        expected = output_qparams.quantize_value(reference)
        previous = output_qparams.quantize_value(
            torch.tanh(hidden_qparams.dequantize_value(previous))
        )

        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
            result = InsertTableOpsPass(program).call(program.graph_module)
            program.validate()

        self.assertTrue(result.modified)
        calls = [node for node in program.graph.nodes if node.op == "call_function"]
        self.assertEqual(calls, [sink])
        self.assertIs(sink.target, exir_ops.backend.tosa.TABLE.default)
        self.assertIs(sink.args[0], source)
        self.assertEqual(sink.meta["input_qparams"], {0: input_qparams})
        self.assertEqual(sink.meta["output_qparams"], {0: output_qparams})
        actual = get_param_tensor(program, sink.args[1]).reshape_as(domain)
        torch.testing.assert_close(actual, expected)
        error = (output_qparams.dequantize_value(actual) - reference).abs()
        previous_error = (output_qparams.dequantize_value(previous) - reference).abs()
        self.assertLess(error.mean(), previous_error.mean())
        self.assertLess(error.max(), previous_error.max())
        self.assertNotIn("constant", program.state_dict)

    def test_int16_chain_builds_one_interpolated_table_and_rescale(self):
        from executorch.backends.arm.tosa.specification import (
            TosaLoweringContext,
            TosaSpecification,
        )

        program, source, first, sink, domain = _quantized_region(dtype=torch.int16)
        input_qparams = first.meta["input_qparams"][0]
        hidden_qparams = first.meta["output_qparams"][0]._replace(scale=0.0005)
        first.meta["output_qparams"][0] = hidden_qparams
        sink.meta["input_qparams"][0] = hidden_qparams
        output_qparams = sink.meta["output_qparams"][0]
        reference = torch.tanh(torch.sigmoid(input_qparams.dequantize_value(domain)))
        hidden = hidden_qparams.quantize_value(
            torch.sigmoid(input_qparams.dequantize_value(domain))
        )
        previous = output_qparams.quantize_value(
            torch.tanh(hidden_qparams.dequantize_value(hidden))
        )
        expected = output_qparams.quantize_value(reference)
        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
        ):
            self.assertTrue(
                InsertTableOpsPass(program).call(program.graph_module).modified
            )
            program.validate()

        tables = [
            node
            for node in program.graph.nodes
            if node.target == exir_ops.backend.tosa.TABLE.default
        ]
        self.assertEqual(len(tables), 1)
        self.assertIs(tables[0].args[0], source)
        table = get_param_tensor(program, tables[0].args[1])
        self.assertEqual(table.dtype, torch.int16)
        self.assertEqual(table.numel(), 513)
        self.assertIs(sink.target, exir_ops.backend.tosa.RESCALE.default)
        self.assertEqual(sink.args[2], [1 / 128])
        index = (domain.to(torch.int32) + 32768) >> 7
        fraction = domain.to(torch.int32) & 127
        interpolated = (
            table[index].to(torch.int32) * (128 - fraction)
            + table[index + 1].to(torch.int32) * fraction
        )
        actual = ((interpolated + 64) >> 7).to(torch.int16)
        self.assertLessEqual(
            int((actual.to(torch.int32) - expected.to(torch.int32)).abs().max()), 2
        )
        error = (output_qparams.dequantize_value(actual) - reference).abs()
        previous_error = (output_qparams.dequantize_value(previous) - reference).abs()
        self.assertLess(error.mean(), previous_error.mean())
        self.assertLess(error.max(), previous_error.max())

    def test_explicit_qdq_region_keeps_internal_values_in_fp32(self):
        for dtype in (torch.int8, torch.int16):
            with self.subTest(dtype=dtype):
                program, _, first, sink, _ = _quantized_region(
                    dtype=dtype, arithmetic=True, scalar_input=True
                )
                input_qparams = first.meta["input_qparams"][0]
                output_qparams = sink.meta["output_qparams"][0]
                info = torch.iinfo(dtype)
                domain = torch.arange(info.min, info.max + 1, dtype=torch.int32).to(
                    dtype
                )
                reference = torch.sigmoid(input_qparams.dequantize_value(domain))
                for op in (torch.add, torch.sub, torch.mul, torch.div):
                    reference = op(reference, torch.tensor(0.25))
                reference = torch.tanh(reference)
                graph = program.graph
                for node in list(graph.nodes):
                    if node.op != "call_function":
                        continue
                    args = list(node.args)
                    with graph.inserting_before(node):
                        for index, qparams in node.meta.pop("input_qparams").items():
                            dequant = graph.call_function(
                                InsertTableOpsPass.dequantize_op,
                                (args[index], *qparams[:5]),
                            )
                            dequant.meta["val"] = torch.empty_like(
                                args[index].meta["val"], dtype=torch.float32
                            )
                            args[index] = dequant
                    node.args = tuple(args)
                    qparams = node.meta.pop("output_qparams")[0]
                    users = list(node.users)
                    with graph.inserting_after(node):
                        quant = graph.call_function(
                            InsertTableOpsPass.quantize_op, (node, *qparams[:5])
                        )
                        quant.meta["val"] = node.meta["val"]
                    node.meta["val"] = torch.empty_like(
                        node.meta["val"], dtype=torch.float32
                    )
                    for user in users:
                        user.replace_input_with(node, quant)
                graph.lint()
                fusion = InsertTableOpsPass(program)
                region = fusion._find_fusion_region(quant)
                self.assertIsNotNone(region)
                self.assertEqual(fusion._region_input_qparams(region), [input_qparams])
                self.assertEqual(
                    fusion._region_output_qparams(region), {0: output_qparams}
                )
                actual = fusion._evaluate_region(
                    region, input_qparams.dequantize_value(domain)
                )
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_single_unary_preserves_input_requantization(self):
        from executorch.backends.arm.tosa.specification import (
            TosaLoweringContext,
            TosaSpecification,
        )

        for target, op, output_scale in (
            (exir_ops.edge.aten.asin.default, torch.asin, 0.02),
            (exir_ops.edge.aten.acos.default, torch.acos, 0.02),
            (exir_ops.edge.aten.atanh.default, torch.atanh, 0.1),
        ):
            with self.subTest(target=target):
                program, source, _, sink, _ = _quantized_region(scalar_input=True)
                input_qparams = QuantArgs(1 / 255, -128, -128, 127, torch.int8)
                activation_qparams = QuantArgs(1 / 128, 0, -128, 127, torch.int8)
                output_qparams = QuantArgs(output_scale, 0, -128, 127, torch.int8)
                graph = program.graph
                with graph.inserting_before(sink):
                    dequant = graph.call_function(
                        InsertTableOpsPass.dequantize_op,
                        (source, *input_qparams[:5]),
                    )
                    dequant.meta["val"] = torch.empty_like(
                        source.meta["val"], dtype=torch.float32
                    )
                    requant = graph.call_function(
                        InsertTableOpsPass.quantize_op,
                        (dequant, *activation_qparams[:5]),
                    )
                    requant.meta["val"] = source.meta["val"]
                sink.target = target
                sink.args = (requant,)
                sink.meta["input_qparams"] = {0: activation_qparams}
                sink.meta["output_qparams"] = {0: output_qparams}
                graph.eliminate_dead_code()
                graph.lint()
                program.graph_module.recompile()

                with TosaLoweringContext(
                    TosaSpecification.create_from_string("TOSA-1.0+INT")
                ):
                    self.assertTrue(
                        InsertTableOpsPass(program).call(program.graph_module).modified
                    )
                    program.validate()

                self.assertIs(sink.target, exir_ops.backend.tosa.TABLE.default)
                table = get_param_tensor(program, sink.args[1])
                input_code = torch.tensor([127], dtype=torch.int8)
                activation_code = activation_qparams.quantize_value(
                    input_qparams.dequantize_value(input_code)
                )
                expected = output_qparams.quantize_value(
                    op(activation_qparams.dequantize_value(activation_code))
                )
                self.assertEqual(
                    table[int(activation_code.item()) + 128].item(),
                    expected.item(),
                )
                self.assertIs(sink.args[0], requant)
                self.assertIs(requant.args[0], dequant)

    def test_pure_int16_requantization_does_not_become_table(self):
        program, source, _, sink, _ = _quantized_region(
            dtype=torch.int16, scalar_input=True
        )
        input_qparams = QuantArgs(1.0, 0, -32768, 32767, torch.int16)
        output_qparams = QuantArgs(0.0006408887, 0, -32768, 32767, torch.int16)
        graph = program.graph
        with graph.inserting_before(sink):
            dequant = graph.call_function(
                InsertTableOpsPass.dequantize_op,
                (source, *input_qparams[:5]),
            )
            dequant.meta["val"] = torch.empty_like(
                source.meta["val"], dtype=torch.float32
            )
        sink.target = InsertTableOpsPass.quantize_op
        sink.args = (dequant, *output_qparams[:5])
        sink.meta.pop("input_qparams")
        sink.meta.pop("output_qparams")
        graph.eliminate_dead_code()
        graph.lint()
        program.graph_module.recompile()

        self.assertFalse(
            InsertTableOpsPass(program).call(program.graph_module).modified
        )
        program.validate()
        self.assertIs(sink.target, InsertTableOpsPass.quantize_op)
        self.assertIs(sink.args[0], dequant)
        self.assertFalse(
            any(
                node.target == exir_ops.backend.tosa.TABLE.default
                for node in program.graph.nodes
            )
        )
        raw = torch.tensor([-21, -11, 18], dtype=torch.int16)
        actual = output_qparams.dequantize_value(
            output_qparams.quantize_value(input_qparams.dequantize_value(raw))
        )
        torch.testing.assert_close(actual, raw.to(torch.float32), rtol=0, atol=0.001)

    def test_single_unary_with_explicit_output_quantization_keeps_input_boundary(self):
        from executorch.backends.arm.tosa.specification import (
            TosaLoweringContext,
            TosaSpecification,
        )

        program, source, _, sink, _ = _quantized_region(scalar_input=True)
        input_qparams = QuantArgs(1 / 255, -128, -128, 127, torch.int8)
        activation_qparams = QuantArgs(1 / 128, 0, -128, 127, torch.int8)
        output_qparams = QuantArgs(0.02, 0, -128, 127, torch.int8)
        graph = program.graph
        with graph.inserting_before(sink):
            dequant = graph.call_function(
                InsertTableOpsPass.dequantize_op,
                (source, *input_qparams[:5]),
            )
            dequant.meta["val"] = torch.empty_like(
                source.meta["val"], dtype=torch.float32
            )
            requant = graph.call_function(
                InsertTableOpsPass.quantize_op,
                (dequant, *activation_qparams[:5]),
            )
            requant.meta["val"] = source.meta["val"]
            activation_input = graph.call_function(
                InsertTableOpsPass.dequantize_op,
                (requant, *activation_qparams[:5]),
            )
            activation_input.meta["val"] = dequant.meta["val"]
            unary = graph.call_function(
                exir_ops.edge.aten.asin.default, (activation_input,)
            )
            unary.meta["val"] = dequant.meta["val"]
        sink.target = InsertTableOpsPass.quantize_op
        sink.args = (unary, *output_qparams[:5])
        sink.meta.pop("input_qparams")
        sink.meta.pop("output_qparams")
        graph.eliminate_dead_code()
        graph.lint()
        program.graph_module.recompile()

        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
            self.assertTrue(
                InsertTableOpsPass(program).call(program.graph_module).modified
            )
            program.validate()

        self.assertIs(sink.target, exir_ops.backend.tosa.TABLE.default)
        self.assertIs(sink.args[0], requant)
        table = get_param_tensor(program, sink.args[1])
        expected = output_qparams.quantize_value(
            torch.asin(
                activation_qparams.dequantize_value(
                    torch.tensor([127], dtype=torch.int8)
                )
            )
        )
        self.assertEqual(table[-1].item(), expected.item())

    def test_per_channel_region_is_rejected(self):
        program, _, first, sink, _ = _quantized_region()
        first.meta["input_qparams"][0] = first.meta["input_qparams"][0]._replace(
            per_channel=True
        )
        sink.meta["output_qparams"][0] = sink.meta["output_qparams"][0]._replace(
            per_channel=True
        )
        self.assertIsNone(InsertTableOpsPass(program)._find_fusion_region(sink))

    def test_int16_input_with_int32_output_keeps_rescale(self):
        from executorch.backends.arm.tosa.specification import (
            TosaLoweringContext,
            TosaSpecification,
        )

        program, _, _, sink, _ = _quantized_region(dtype=torch.int16)
        sink.meta["output_qparams"][0] = QuantArgs(
            1e-8, 0, -2147483648, 2147483647, torch.int32
        )
        sink.meta["val"] = torch.empty_like(sink.meta["val"], dtype=torch.int32)
        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
        ):
            assert InsertTableOpsPass(program).call(program.graph_module).modified
            program.validate()

        self.assertIs(sink.target, exir_ops.backend.tosa.RESCALE.default)
        self.assertIs(sink.args[1], torch.int32)
        self.assertGreater(sink.args[2][0], 1)
        table_op = sink.args[0]
        table = get_param_tensor(program, table_op.args[1])
        self.assertEqual(table.dtype, torch.int16)
        self.assertEqual(table.numel(), 513)

    def test_arm_quantized_chain_lowers_to_one_table(self):
        from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
        from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
        from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
        from executorch.backends.arm.quantizer import (
            get_symmetric_quantization_config,
            TOSAQuantizer,
        )
        from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
        from executorch.backends.arm.tosa.specification import TosaLoweringContext
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        x = torch.randn(2, 5, generator=torch.Generator().manual_seed(42))
        spec = TosaCompileSpec("TOSA-1.0+INT")
        quantizer = TOSAQuantizer(spec)
        quantizer.set_global(get_symmetric_quantization_config())
        prepared = prepare_pt2e(export(_SigmoidTanh(), (x,)).module(), quantizer)
        prepared(x)
        quantized = convert_pt2e(prepared)
        program = to_edge(export(quantized, (x,))).exported_program()
        manager = ArmPassManager(spec)
        manager.add_pass(FoldAndAnnotateQParamsPass(program))
        manager.add_pass(InsertTableOpsPass(program))

        with TosaLoweringContext(spec.tosa_spec):
            manager._transform(program, program.graph_module)
            program.validate()

        self.assertEqual(
            sum(
                node.op == "call_function"
                and node.target == exir_ops.backend.tosa.TABLE.default
                for node in program.graph.nodes
            ),
            1,
        )
        self.assertEqual(len(program.state_dict), 1)


if __name__ == "__main__":
    unittest.main()
