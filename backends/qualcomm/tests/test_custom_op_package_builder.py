# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import importlib
import unittest
from unittest.mock import patch

from executorch.backends.qualcomm.custom_op import interface


class TestCustomOpPackageBuilderImports(unittest.TestCase):
    def test_import_without_sdk(self):
        with patch.dict(
            "sys.modules", {"qti.aisw.op_package_generator.generator": None}
        ):
            module = importlib.reload(interface)
            with self.assertRaises(ImportError) as caught:
                module.QnnCustomOpPackageBuilder("unused.xml", {})

        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)
        self.assertIn("QnnOpPackageGenerator", str(caught.exception))

    def test_incompatible_sdk_error_is_preserved(self):
        sdk_error = ImportError("libpython3.10.so.1.0: cannot open shared object file")
        original_import = builtins.__import__

        def import_with_incompatible_sdk(name, *args, **kwargs):
            if name == "qti.aisw.op_package_generator.generator":
                raise sdk_error
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_with_incompatible_sdk):
            module = importlib.reload(interface)
            with self.assertRaises(ImportError) as caught:
                module.QnnCustomOpPackageBuilder("unused.xml", {})

        self.assertIs(caught.exception.__cause__, sdk_error)
        self.assertIn(str(sdk_error), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
