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
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.models.yolo26 import YOLO26Model


class TestMilestoneYolo26(unittest.TestCase):
    def test_yolo26_fp16(self):
        torch.manual_seed(8)
        model = YOLO26Model().get_eager_model()
        example_input = YOLO26Model().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (tester.export().to_edge_transform_and_lower().to_executorch())
