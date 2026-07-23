# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import re
import unittest


def _conditional_branches(source: str, condition: str) -> tuple[str, str]:
    lines = source.splitlines()
    condition_pattern = re.compile(
        rf"^\s*if\s*\(\s*{re.escape(condition)}\s*\)\s*$",
        re.IGNORECASE,
    )
    command_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    start = next(
        (index for index, line in enumerate(lines) if condition_pattern.match(line)),
        None,
    )
    if start is None:
        raise AssertionError(f"if({condition}) branch not found")

    depth = 1
    else_index = None
    for index in range(start + 1, len(lines)):
        match = command_pattern.match(lines[index])
        if match is None:
            continue
        command = match.group(1).lower()
        if command == "if":
            depth += 1
        elif command == "endif":
            depth -= 1
            if depth == 0:
                if else_index is None:
                    raise AssertionError(f"if({condition}) has no else() branch")
                return (
                    "\n".join(lines[start + 1 : else_index]),
                    "\n".join(lines[else_index + 1 : index]),
                )
        elif command == "else" and depth == 1:
            if else_index is not None:
                raise AssertionError(f"if({condition}) has multiple else() branches")
            else_index = index
    raise AssertionError(f"if({condition}) has no matching endif()")


class TestCMakeConfiguration(unittest.TestCase):
    def test_branch_parser_keeps_nested_conditionals_in_native_branch(self) -> None:
        source = """
if ( EMSCRIPTEN )
  wasm_command()
else()
  if(APPLE)
    apple_command()
  else()
    linux_command()
  endif()
endif()
"""
        wasm_branch, native_branch = _conditional_branches(source, "EMSCRIPTEN")

        self.assertIn("wasm_command()", wasm_branch)
        self.assertIn("apple_command()", native_branch)
        self.assertIn("linux_command()", native_branch)

    def test_emscripten_uses_port_instead_of_native_dawn(self) -> None:
        cmake = pathlib.Path(__file__).parents[1] / "CMakeLists.txt"
        wasm_branch, native_branch = _conditional_branches(
            cmake.read_text(), "EMSCRIPTEN"
        )
        port_flag = r'"--use-port=emdawnwebgpu"'

        self.assertRegex(
            wasm_branch,
            rf"target_compile_options\s*\(\s*webgpu_backend\s+PUBLIC\s+{port_flag}\s*\)",
        )
        self.assertRegex(
            wasm_branch,
            rf"target_link_options\s*\(\s*webgpu_backend\s+PUBLIC\s+{port_flag}\s*\)",
        )
        self.assertNotIn("find_package(Dawn", wasm_branch)
        self.assertNotIn("--use-port=emdawnwebgpu", native_branch)
        self.assertRegex(
            native_branch, r"find_package\s*\(\s*Dawn\s+REQUIRED\s*\)"
        )
