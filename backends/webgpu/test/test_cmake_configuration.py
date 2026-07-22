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
else()"""

        self.assertIn(wasm_branch, source)
        branch_start = source.index(wasm_branch)
        branch_end = source.index("\nendif()", branch_start)
        self.assertIn("find_package(Dawn REQUIRED)", source[branch_start:branch_end])


if __name__ == "__main__":
    unittest.main()
