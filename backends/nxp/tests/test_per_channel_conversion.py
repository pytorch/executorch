# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import (
    act_qspec,
    NeutronAtenQuantizer,
    wgt_qspec,
)
from executorch.backends.nxp.quantizer.patterns import (
    NodeArgsIdx,
    PartitionAnchors,
    QuantizationPattern,
)
from executorch.backends.nxp.quantizer.utils import get_bias_qparams
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import Conv2dModule
from executorch.exir.dialects._ops import ops as exir_ops

from torch import fx
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationConfig,
    QuantizationSpec,
)


class Conv2dPatternPerChannel(QuantizationPattern):

    def __init__(self, is_per_channel: bool):
        super().__init__()
        self.is_per_channel = is_per_channel

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv2d_node = fused_partition[0].nodes[-1]

        bias_qscheme = (
            torch.per_channel_symmetric
            if self.is_per_channel
            else torch.per_tensor_symmetric
        )
        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv2d_node.args[0], conv2d_node),
                (conv2d_node.args[1], conv2d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2**31) + 1,
            quant_max=2**31 - 1,
            qscheme=bias_qscheme,
            ch_axis=0,
        )

        weight_qscheme = (
            torch.per_channel_symmetric
            if self.is_per_channel
            else torch.per_tensor_symmetric
        )
        weight_observer_or_fake_quant_ctr = (
            PerChannelMinMaxObserver if self.is_per_channel else MinMaxObserver
        )
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=weight_qscheme,
            ch_axis=0,
        )

        return PartitionAnchors(
            inputs=[(conv2d_node, NodeArgsIdx(0))],
            weights=[(conv2d_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=[(conv2d_node, NodeArgsIdx(2), bias_quantization_qspec)],
            output=[(conv2d_node,)],
        )


class TestPerChannelConversion(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(25)
        np.random.seed(25)

    def test_per_channel_convolution(self):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            model = Conv2dModule(
                in_channels=8, out_channels=32, kernel_size=5, padding=3
            )
            input_shape = (1, 8, 32, 32)

            static_qconfig = QuantizationConfig(act_qspec, act_qspec, wgt_qspec, None)
            _ = to_quantized_edge_program(
                model,
                input_shape,
                get_quantizer_fn=lambda: NeutronAtenQuantizer(
                    Conv2dPatternPerChannel(is_per_channel=True), static_qconfig
                ),
                use_neutron_for_format_conversion=False,
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )

            convert_run_compare(
                exported_program,
                tflite_input_preprocess=ToChannelLastPreprocess(),
                tfl_model=tflite_flatbuffers_model,
                tflite_output_preprocess=ToChannelFirstPreprocess(),
                input_data=input_data,
                atol=1.0,
            )

            nodes = list(exported_program.graph.nodes)

            assert (
                nodes[8].target
                == exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
            )
            assert (
                nodes[9].target
                == exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
            )
            assert nodes[10].target == exir_ops.edge.aten.convolution.default
