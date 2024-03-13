# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

kernel_mode = None  # either aten mode or portable mode
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    kernel_mode = "portable"
except Exception:
    print("can't load portable lib")

try:
    from executorch.extension.pybindings.aten_lib import (  # noqa: F811
        _load_for_executorch_from_buffer,
    )

    assert kernel_mode is None

    kernel_mode = "aten"
except Exception:
    print("can't load aten lib")

assert kernel_mode is not None


from executorch.extension.pybindings.test.make_test import make_test


class PybindingsTest(unittest.TestCase):
    def test(self):
        make_test(self, _load_for_executorch_from_buffer)(self)
