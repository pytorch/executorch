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
from executorch.examples.models.mobilenet_v3 import MV3Model


class Test_Milestone_MobilenetV3(unittest.TestCase):
    def test_mv3_fp16(self):
        model = MV3Model().get_eager_model()
        example_input = MV3Model().get_example_inputs()

        (
            SamsungTester(model,
                          example_input,
                          [gen_samsung_backend_compile_spec("E9955")],)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(atol=0.06, rtol=0.06)
            # TODO: theshold value should be updated after fixing accuracy issue
        )
