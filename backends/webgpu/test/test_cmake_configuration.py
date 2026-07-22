# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import unittest


class TestCMakeConfiguration(unittest.TestCase):
    def test_emscripten_uses_port_instead_of_native_dawn(self) -> None:
        cmake = pathlib.Path(__file__).parents[1] / "CMakeLists.txt"
        source = cmake.read_text()
        wasm_branch = """if(EMSCRIPTEN)
  target_compile_options(webgpu_backend PUBLIC \"--use-port=emdawnwebgpu\")
  target_link_options(webgpu_backend PUBLIC \"--use-port=emdawnwebgpu\")
else()"""

        self.assertIn(wasm_branch, source)
        # Dawn is linked only in the native (else) branch, never under EMSCRIPTEN.
        # Split on the branch head so the assertion does not depend on which
        # endif() (the nested APPLE one vs the outer one) appears first.
        native_branch = source.split(wasm_branch, 1)[1]
        self.assertIn("find_package(Dawn REQUIRED)", native_branch)
