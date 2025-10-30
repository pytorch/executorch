# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple, Union

import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.tester import Quantize as XNNPackQuantize, Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)

from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

try:
    import executorch.extension.pybindings.portable_lib  # noqa[F401]
    import executorch.kernels.quantized  # noqa[F401]

    has_quantized_ops = True
except:
    has_quantized_ops = False
    print("Missing quantized ops")


class TestPropagateCustomMetaPass(unittest.TestCase):
    class ModuleLinear(torch.nn.Module):
        def __init__(
            self,
            in_size: int = 2,
            input_channels: int = 4,
            output_channels: int = 4,
            dtype: torch.dtype = torch.float,
            use_bias: bool = False,
        ):
            super().__init__()
            self.linear = torch.nn.Linear(
                input_channels, output_channels, bias=use_bias
            ).to(dtype=dtype)

            self.ic = input_channels
            self.oc = output_channels
            assert dtype in [torch.float, torch.half], "Unsupported op dtype"
            self.op_dtype = dtype
            self.in_size = in_size

        def forward(self, x: torch.Tensor):
            return self.linear(x)

        def get_random_inputs(self):
            inp = torch.randn(self.in_size, self.ic).to(self.op_dtype)
            return (inp,)

    class Export(BaseStages.Export):
        def run(
            self,
            artifact: torch.nn.Module,
            inputs: Tuple[torch.Tensor],
        ) -> None:

            tagged_module = torch.export.export(
                artifact, inputs, dynamic_shapes=self.dynamic_shapes, strict=True
            ).module()
            delegate_external_constants_pass_unlifted(
                module=tagged_module,
                gen_tag_fn=lambda x: "model",  # This is the filename the weights will be saved to. In this case, weights will be saved as "model.ptd"
            )
            self.exported_program = torch.export.export(
                tagged_module, inputs, dynamic_shapes=self.dynamic_shapes, strict=True
            )

    def _test_linear(
        self,
        partitioner: XnnpackPartitioner,
        quantization_stage: Union[BaseStages.Quantize, BaseStages.Quantize_],
    ):
        eager_model = self.ModuleLinear(
            in_size=1,
            input_channels=32,
            output_channels=2,
        )
        test_inputs = eager_model.get_random_inputs()

        tester = Tester(eager_model, test_inputs)
        tester.quantize(quantization_stage)
        tester.export(self.Export())
        tester.to_edge_transform_and_lower(
            ToEdgeTransformAndLower([partitioner])
        ).to_executorch()
        tester.run_method_and_compare_outputs()

        exec = tester.get_artifact()
        program_buffer = exec.buffer
        self.assertEqual(len(exec._tensor_data), 1)
        data_buffer = bytes(exec._tensor_data.pop("model"))
        self.assertTrue(len(data_buffer) > 200)
        from executorch.extension.pybindings import portable_lib as runtime

        module = runtime._load_for_executorch_from_buffer(program_buffer, data_buffer)
        output = module.forward(test_inputs)
        reference_output = exec.exported_program().module()(
            test_inputs[0],
        )
        self.assertTrue(torch.allclose(output[0], reference_output, 1e-2))

        # with self.assertRaises(RuntimeError):
        #     runtime._load_for_executorch_from_buffer(program_buffer).forward(
        #         test_inputs
        #     )

    def test_quantize_(self):
        # Quantize with torchao quantize_ API.
        DynamicallyQuantizedPartitioner = XnnpackPartitioner(
            config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
            per_op_mode=False,
        )
        linear_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        self._test_linear(
            DynamicallyQuantizedPartitioner, BaseStages.Quantize_(config=linear_config)
        )

    def test_pt2e_quantize(self):
        # Quantize with pt2e quantize.
        quant_configs = [
            # per_tensor
            get_symmetric_quantization_config(is_per_channel=False, is_dynamic=False),
            # per_channel
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False),
            # per_channel_dynamic
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True),
        ]
        for quant_config in quant_configs:
            precision = (
                ConfigPrecisionType.DYNAMIC_QUANT
                if quant_config.input_activation.is_dynamic
                else ConfigPrecisionType.STATIC_QUANT
            )
            for per_op_mode in [True, False]:
                partitioner = XnnpackPartitioner(
                    config_precisions=precision, per_op_mode=per_op_mode
                )
                self._test_linear(
                    partitioner, XNNPackQuantize(quantization_config=quant_config)
                )
