# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torchvision
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import QConfigMapping  # @manual
from torch.ao.quantization.backend_config import get_executorch_backend_config
from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.ao.quantization.quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export
from torch.testing import FileCheck
from torch.testing._internal.common_quantized import override_quantized_engine

# load executorch out variant ops
torch.ops.load_library("//executorch/kernels/quantized:custom_ops_generated_lib")


class TestQuantization(unittest.TestCase):
    """prepare_pt2e and convert_pt2e are OSS APIs, the rest are all meta-only

    APIs for now, but we plan to open source them in the future
    """

    def test_resnet(self) -> None:
        import copy

        with override_quantized_engine("qnnpack"):
            torch.backends.quantized.engine = "qnnpack"
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m = torch._export.capture_pre_autograd_graph(
                m, copy.deepcopy(example_inputs)
            )

            quantizer = XNNPACKQuantizer()
            operator_config = get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(operator_config)
            m = prepare_pt2e(m, quantizer)  # pyre-fixme[6]
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            after_prepare_result = m(*example_inputs)[0]
            m = convert_pt2e(m)

            # TODO: conv, conv_relu, linear delegation
            # quantized ops to implement: add_relu
            compile_config = EdgeCompileConfig(
                _check_ir_validity=False,
            )
            m = to_edge(
                export(m, example_inputs), compile_config=compile_config
            ).transform([QuantFusionPass(), SpecPropPass()])

            after_quant_result = m.exported_program().module()(*example_inputs)[0]
            FileCheck().check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor"
            ).check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor"
            ).run(
                m.exported_program().graph_module.code
            )
            # after_quant_fusion_result = m(*example_inputs)[0]

            # TODO: implement torch.ops.quantized_decomposed.add_relu.out
            # m = m.to_executorch().dump_graph_module()
            # after_to_executorch = m(*example_inputs)[0]
            # test the result before and after to_executorch matches
            # TODO: debug why this is a mismatch
            # self.assertTrue(torch.equal(after_quant_fusion_result, after_to_executorch))
            # self.assertEqual(compute_sqnr(after_quant_fusion_result, after_to_executorch), torch.tensor(float("inf")))

            # comparing with existing fx graph mode quantization reference flow
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_executorch_backend_config()
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(
                m_fx, backend_config=backend_config
            )
            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            self.assertTrue(
                torch.allclose(after_prepare_result, after_prepare_result_fx, atol=1e-6)
            )

            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(
                torch.max(after_quant_result - after_quant_result_fx) < 1e-1
            )
            self.assertTrue(
                compute_sqnr(after_quant_result, after_quant_result_fx) > 35
            )
