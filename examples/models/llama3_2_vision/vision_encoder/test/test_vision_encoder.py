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
    VisionEncoderConfig,
)
from torch._inductor.package import load_package, package_aoti


class FlamingoVisionEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flamingo_vision_encoder(self) -> None:
        model = FlamingoVisionEncoderModel(VisionEncoderConfig())
        encoder = model.model
        eager_res = encoder.forward(*model.get_example_inputs())

        # AOTI
        so = torch._export.aot_compile(
            encoder,
            model.get_example_inputs(),
            options={"aot_inductor.package": True},
            dynamic_shapes=model.get_dynamic_shapes(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = package_aoti(os.path.join(tmpdir, "vision_encoder.pt2"), so)
            print(path)
            encoder_aoti = load_package(path)

            y = encoder_aoti(*model.get_example_inputs())

            self.assertTrue(torch.allclose(y, eager_res))
