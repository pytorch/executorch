# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.
import unittest

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet50Model


class TestMilestoneDeepLabV3(unittest.TestCase):
    @unittest.skip("Need to be fixed issue")
    def test_dl3_fp16(self):
        model = DeepLabV3ResNet50Model().get_eager_model()
        example_input = DeepLabV3ResNet50Model().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.009)
        )
