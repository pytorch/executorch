# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

from executorch.frontends.fused_quant.quick_start import main


class QuickStartTest(unittest.TestCase):
    def test_saves_a_program_that_tracks_the_float_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fused_quant_portable.pte")

            error = main(path)

            self.assertGreater(os.path.getsize(path), 0)
        self.assertLess(error, 0.1)
