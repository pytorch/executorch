import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.utils.utils import get_param_tensor
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestCheckQuantParams(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def create_invalid_value_injector(
        self, invalid_value, is_per_channel=False, is_zp=False
    ):
        def inject_invalid_scale_in_per_tensor(aten):
            for node in aten.graph_module.graph.nodes:
                target_to_find = (
                    torch.ops.quantized_decomposed.quantize_per_tensor.default
                    if not is_per_channel
                    else torch.ops.quantized_decomposed.dequantize_per_channel.default
                )
                if node.op == "call_function" and node.target == target_to_find:
                    if is_zp:
                        node_args = list(node.args)
                        node_args[2] = invalid_value
                        node.args = tuple(node_args)
                        break
                    else:
                        scale = node.args[1]
                        if is_per_channel:
                            self.assertTrue(isinstance(scale, torch.fx.Node))
                            scale_tensor = get_param_tensor(aten, scale)
                            scale_tensor[2] = invalid_value
                        else:
                            self.assertTrue(isinstance(scale, float))
                            node_args = list(node.args)
                            node_args[1] = invalid_value
                            node.args = tuple(node_args)
                            break

        return inject_invalid_scale_in_per_tensor

    def _test_check_quant_message(self, ep_modifier, expected_message):
        torch._dynamo.reset()
        mod = torch.nn.Linear(10, 10)
        quantizer = XNNPACKQuantizer()
        captured = export(mod, (torch.randn(1, 10),), strict=True).module()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(captured, quantizer)

        prepared(*(torch.randn(1, 10),))
        converted = convert_pt2e(prepared)
        aten = torch.export.export(converted, (torch.randn(1, 10),))

        ep_modifier(aten)

        with self.assertRaises(ValueError) as context:
            to_edge_transform_and_lower(aten, partitioner=[XnnpackPartitioner()])

        self.assertEqual(str(context.exception), expected_message)

    def test_in_per_tensor_quant(self):
        for invalid_scale in [
            float("nan"),
            float("inf"),
            -float("inf"),
            1.0000002153053333e-39,
        ]:
            self._test_check_quant_message(
                self.create_invalid_value_injector(invalid_scale),
                "Invalid quantization scale or zero point for quantized_decomposed_quantize_per_tensor_default: "
                "Scales must be finite and normal, however found scale value: "
                f"{invalid_scale} in scale tensor at index: (0,)",
            )

    def test_in_per_channel_quant(self):
        for invalid_scale in [
            float("nan"),
            float("inf"),
            -float("inf"),
            1.0000002153053333e-39,
        ]:
            self._test_check_quant_message(
                self.create_invalid_value_injector(invalid_scale, is_per_channel=True),
                "Invalid quantization scale or zero point for quantized_decomposed_dequantize_per_channel_default: "
                "Scales must be finite and normal, however found scale value: "
                f"{invalid_scale} in scale tensor at index: (2,)",
            )

    def test_inject_invalid_zp(self):
        for invalid_zp in [-129, 128]:
            self._test_check_quant_message(
                self.create_invalid_value_injector(
                    invalid_zp, is_zp=True, is_per_channel=False
                ),
                "Invalid quantization scale or zero point for quantized_decomposed_quantize_per_tensor_default: "
                f"Found invalid zeropoint {invalid_zp} "
                "in zero point tensor at index: (0,)",
            )
