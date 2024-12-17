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


class FlamingoVisionEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flamingo_vision_encoder(self) -> None:
        model = FlamingoVisionEncoderModel()
        encoder = model.model
        eager_res = encoder.forward(*model.get_example_inputs())

        # AOTI
        ep = torch.export.export(
            encoder,
            model.get_example_inputs(),
            dynamic_shapes=model.get_dynamic_shapes(),
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
