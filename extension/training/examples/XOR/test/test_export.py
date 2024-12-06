# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

from executorch.extension.training.examples.XOR.export_model import _export_model


class TestXORExport(unittest.TestCase):
    def test(self):
        _ = _export_model()
        # Expect that we reach this far without an exception being thrown.
        self.assertTrue(True)
