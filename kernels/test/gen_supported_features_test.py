#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
"""
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
"""
        )
        result = generate_definition(y)
        self.assertTrue(".output_resize = true," in result)
        self.assertTrue(".op_gelu_dtype_double = true," in result)


if __name__ == "__main__":
    unittest.main()
