# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.test.tester import Tester
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.passes.quantize_io_pass import (
    get_config_method_name,
    QuantizeInputs,
)
from executorch.exir.tensor import dim_order_from_stride
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestIODimOrder(unittest.TestCase):
    """Delegate inputs and outputs whose dim order is neither contiguous nor NHWC."""

    def setUp(self):
        torch._dynamo.reset()

    class BMM(torch.nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class AddConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x, y):
            return y + self.conv(x)

    @staticmethod
    def _dim_order(tensor):
        # The order the runtime uses, which breaks size-1 ties unlike dim_order()
        return tuple(dim_order_from_stride(tensor.stride(), tuple(tensor.shape)))

    def _test(self, module, inputs, dynamic_shapes=None):
        # The harness only generates contiguous or channels-last random inputs,
        # so each run's inputs are given explicitly.
        tester = Tester(module, inputs[0], dynamic_shapes=dynamic_shapes)
        tester.export().to_edge_transform_and_lower().check_count(
            {"torch.ops.higher_order.executorch_call_delegate": 1}
        ).to_executorch().serialize()
        for run_inputs in inputs:
            tester.run_method_and_compare_outputs(inputs=run_inputs)

    @staticmethod
    def _permuted(n=8):
        # Dim order (2, 0, 1), which is not its own inverse, so a transpose in
        # the wrong direction fails.
        return torch.randn(16, 4, n).permute(1, 2, 0)

    def test_fp32_bmm_permuted_input(self):
        x = self._permuted()
        self.assertEqual(x.dim_order(), (2, 0, 1))
        self._test(self.BMM(), [(x, torch.randn(4, 16, 3))])

    def test_fp32_bmm_permuted_input_dynamic(self):
        n = torch.export.Dim("n", min=2, max=8)
        self._test(
            self.BMM(),
            [(self._permuted(size), torch.randn(4, 16, 3)) for size in (8, 2, 5)],
            dynamic_shapes=({1: n}, None),
        )

    def test_fp32_add_transposed_input_and_output(self):
        x, y = torch.randn(1, 8, 4, 16).transpose(1, 2), torch.randn(1, 4, 8, 16)
        self.assertEqual(self._dim_order(x), (0, 2, 1, 3))
        self.assertEqual(self._dim_order(self.Add()(x, y)), (0, 2, 1, 3))
        self._test(self.Add(), [(x, y)])

    def test_fp32_add_transposed_size_one_dims(self):
        x = torch.randn(1, 8, 1, 16).transpose(1, 2)
        self.assertEqual(x.dim_order(), (0, 2, 1, 3))
        self.assertEqual(self._dim_order(x), (0, 1, 2, 3))
        self._test(self.Add(), [(x, torch.randn(1, 1, 8, 16))])

    def test_fp32_conv_feeds_transposed_output(self):
        # The add is NHWC after the conv, so the XNNPACK passes copy it back to
        # contiguous before the output, while the runtime keeps y's dim order.
        module = self.AddConv().eval()
        x, y = torch.randn(1, 3, 8, 16), torch.randn(1, 8, 8, 16).transpose(1, 2)
        self.assertEqual(self._dim_order(module(x, y)), (0, 2, 1, 3))
        self._test(module, [(x, y)])

    def test_fp32_add_channels_last_3d(self):
        x = torch.randn(1, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)
        self._test(self.Add(), [(x, torch.randn(1, 3, 4, 5, 6))])

    def test_qs8_add_quantized_permuted_input(self):
        torch.manual_seed(0)
        x, y = self._permuted(), torch.randn(4, 8, 16)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config())
        module = prepare_pt2e(
            torch.export.export(self.Add().eval(), (x, y), strict=True).module(),
            quantizer,
        )
        module(x, y)
        module = convert_pt2e(module)
        edge = to_edge(
            torch.export.export(module, (x, y), strict=True),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        # Take x already quantized, so the delegate input itself is quantized.
        edge = edge.transform(
            [QuantizeInputs(edge, quantized_inputs_idx=[0], method_name="forward")]
        )
        program = edge.to_backend(XnnpackPartitioner()).to_executorch()
        graph = program.exported_program().graph
        delegates = [
            n
            for n in graph.nodes
            if n.target == torch.ops.higher_order.executorch_call_delegate
        ]
        self.assertEqual(len(delegates), 1)
        x_node = next(n for n in graph.nodes if n.op == "placeholder")
        self.assertIn(x_node, delegates[0].args)
        self.assertEqual(x_node.meta["val"].dtype, torch.int8)

        qparams = edge._config_methods
        quantized_x = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x,
            *(
                qparams[get_config_method_name("forward", "input", 0, key)]
                for key in ("scale", "zp", "quant_min", "quant_max")
            ),
            torch.int8,
        )
        out = _load_for_executorch_from_buffer(program.buffer).run_method(
            "forward", (quantized_x, y)
        )[0]
        # XNNPACK may round the requantized sum one step away from the reference.
        output_scale = next(
            n.args[1]
            for n in reversed(module.graph.nodes)
            if n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        torch.testing.assert_close(out, module(x, y), atol=1.5 * output_scale, rtol=0)
