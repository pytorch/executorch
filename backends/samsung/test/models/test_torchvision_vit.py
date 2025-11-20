# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.
import unittest

import torch
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.examples.models.torchvision_vit import TorchVisionViTModel


class TestMilestoneTorchVisionViT(unittest.TestCase):
    @unittest.skip("Need to be fixed")
    def test_torchvision_vit_fp16(self):
        torch.manual_seed(8)
        model = TorchVisionViTModel().get_eager_model()
        example_input = TorchVisionViTModel().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(
                inputs=example_input, atol=0.005, rtol=0.005
            )
        )
