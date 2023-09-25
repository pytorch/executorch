#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import yaml
from executorch.kernels.test.gen_supported_features import (
    generate_definition,
    generate_header,
)


class TestGenSupportedFeatures(unittest.TestCase):
    def test_gen_header(self):
        y = yaml.load(
            """
- namespace: global
  is_aten:
    type: bool
    default: false
  output_resize:
    type: bool
    default: true

- namespace: op_gelu
  dtype_double:
    type: bool
    default: true
""",
            Loader=yaml.FullLoader,
        )
        result = generate_header(y)
        self.assertTrue("bool is_aten = false;" in result)
        self.assertTrue("bool output_resize = true;" in result)
        self.assertTrue("bool op_gelu_dtype_double = true;" in result)

    def test_gen_def(self):
        y = yaml.load(
            """
- namespace: global
  output_resize: true

- namespace: op_gelu
  dtype_double: true
""",
            Loader=yaml.FullLoader,
        )
        result = generate_definition(y)
        self.assertTrue(".output_resize = true," in result)
        self.assertTrue(".op_gelu_dtype_double = true," in result)


if __name__ == "__main__":
    unittest.main()
