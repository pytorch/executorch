# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

kernel_mode = None  # either aten mode or portable mode
try:
    from executorch.extension.pybindings import portable_lib as runtime

    kernel_mode = "portable"
except Exception:
    print("can't load portable lib")

if kernel_mode is None:
    try:
        from executorch.extension.pybindings import aten_lib as runtime  # noqa: F811

        kernel_mode = "aten"
    except Exception:
        print("can't load aten lib")

assert kernel_mode is not None


from executorch.extension.pybindings.test.make_test import make_test


class PybindingsTest(unittest.TestCase):
    def test(self):
        make_test(self, runtime)(self)
