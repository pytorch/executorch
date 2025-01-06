# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Export and ExecuTorch tests for CLIP vision encoder are covered by test_models.sh.
# Only test AOTI in this file
import os
import tempfile
import unittest

import torch

from executorch.examples.models.llama3_2_vision.vision_encoder import (
    FlamingoVisionEncoderModel,
)
from torch.testing import assert_close
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeCompileConfig
from torch._inductor.package import package_aoti
from torch.nn.attention import SDPBackend
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                    XnnpackDynamicallyQuantizedPartitioner,
)

class FlamingoVisionEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flamingo_vision_encoder_et(self) -> None:
        with torch.no_grad():
            vision_model = FlamingoVisionEncoderModel(enable_source_transforms=False)
            encoder_no_source_transform_outputs = vision_model.model.forward(*vision_model.get_example_inputs())
            vision_model.source_transofrm()
            encoder = vision_model.model
            encoder_source_transform_outputs = encoder.forward(*vision_model.get_example_inputs())
            assert_close(encoder_source_transform_outputs, encoder_no_source_transform_outputs)

            with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
                training_output = torch.export.export_for_training(encoder, vision_model.get_example_inputs(), dynamic_shapes=vision_model.get_dynamic_shapes())
                assert_close(encoder(*vision_model.get_example_inputs()), training_output.module()(*vision_model.get_example_inputs()))

                dynamic_quantizer = XNNPACKQuantizer()
                operator_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                dynamic_quantizer.set_global(operator_config_dynamic)
                prepare = prepare_pt2e(training_output.module(), dynamic_quantizer)
                prepare(*vision_model.get_example_inputs())
                convert = convert_pt2e(prepare)
                DuplicateDynamicQuantChainPass()(convert)

                export_output = torch.export.export(convert, vision_model.get_example_inputs(), dynamic_shapes=vision_model.get_dynamic_shapes())

                edge = to_edge_transform_and_lower(export_output, partitioner=[
                    XnnpackDynamicallyQuantizedPartitioner(),
                ], compile_config=EdgeCompileConfig(_check_ir_validity=False))
                edge.to_executorch()

    def test_flamingo_vision_encoder_aoti(self) -> None:
        model = FlamingoVisionEncoderModel()
        encoder = model.model
        eager_res = encoder.forward(*model.get_example_inputs())

        # AOTI
        ep = torch.export.export(
            encoder,
            model.get_example_inputs(),
            dynamic_shapes=model.get_dynamic_shapes(),
            strict=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = torch._inductor.aoti_compile_and_package(
                ep,
                package_path=os.path.join(tmpdir, "vision_encoder.pt2"),
            )
            print(path)
            encoder_aoti = torch._inductor.aoti_load_package(path)

            y = encoder_aoti(*model.get_example_inputs())
        assert_close(y, eager_res, rtol=1e-4, atol=1e-4)
