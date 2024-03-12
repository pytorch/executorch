# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import executorch.exir as exir
import executorch.exir.tests.models as models
import torch

from parameterized import parameterized


class TestCapture(unittest.TestCase):
    # pyre-ignore
    @parameterized.expand(models.MODELS)
    def test_module_call(self, model_name: str, model: torch.nn.Module) -> None:
        inputs = model.get_random_inputs()
        expected = model(*inputs)
        # TODO(ycao): Replace it with capture_multiple
        exported_program = exir.capture(model, inputs, exir.CaptureConfig())

        self.assertTrue(torch.allclose(expected, exported_program(*inputs)))
