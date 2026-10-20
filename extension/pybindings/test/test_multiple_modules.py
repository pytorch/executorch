# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest


class MultiplePybindingsModulesTest(unittest.TestCase):
    def test_result_types_are_module_local(self) -> None:
        if importlib.util.find_spec("executorch.extension.pybindings.core") is None:
            self.skipTest("core pybindings are only built by internal Buck targets")

        from executorch.extension.pybindings import core, portable_lib

        self.assertIsNot(core.ExecuTorchResult, portable_lib.ExecuTorchResult)


if __name__ == "__main__":
    unittest.main()
