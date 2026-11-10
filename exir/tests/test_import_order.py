# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import subprocess
import sys
import unittest

_IMPORT_WITHOUT_EXIR_INIT = """
import importlib, importlib.util, sys, types

spec = importlib.util.find_spec("executorch.exir")
exir = types.ModuleType("executorch.exir")
exir.__path__ = spec.submodule_search_locations
sys.modules["executorch.exir"] = exir
importlib.import_module(sys.argv[1])
"""


class TestImportOrder(unittest.TestCase):
    def test_import_without_exir_init(self) -> None:
        # Lazy imports can defer executorch/exir/__init__.py, so each of these
        # must import cleanly as the first executorch.exir module.
        for module in (
            "executorch.exir.verification.verifier",
            "executorch.exir.program",
            "executorch.exir._serialize._serialize",
            "executorch.exir.backend.backend_api",
        ):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c", _IMPORT_WITHOUT_EXIR_INIT, module],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
