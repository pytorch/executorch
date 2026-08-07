# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Source contract for the op-test driver's typed input loader.

`op_test_driver.cpp` loads each manifest input by its declared dtype -- `bool` via
`load_int8_bin` + `ScalarType::Bool`, `int32` via `load_int32_bin`, everything else
via `load_fp32_bin`. Deleting the BOOL branch, reordering it after the fp32
fallback, or widening a bool input to fp32 would silently change what every
bool-input case (`where`, `to_copy_bool_input_to_float`) actually feeds the GPU,
and the golden comparison would still "pass" on the wrong bytes.

These checks read source, not artifacts: they need no torch and no GPU, so they run
in any environment. The producer side (`generate_op_tests.py`) is pinned to the same
three dtype strings, and `cases.py` is pinned to wire at least one input generator
per dtype -- i.e. a generated manifest carries at least one entry of each.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

_OP_TESTS_DIR: Path = Path(__file__).resolve().parent
_OPS_DIR: Path = _OP_TESTS_DIR.parent / "ops"

_DRIVER: Path = _OP_TESTS_DIR / "op_test_driver.cpp"
_GENERATOR: Path = _OP_TESTS_DIR / "generate_op_tests.py"
_CASES: Path = _OP_TESTS_DIR / "cases.py"

_INPUT_LOOP_START: str = "for (const auto& in : e_.inputs) {"
_INPUT_LOOP_END: str = "std::vector<EValue> inputs;"
_BOOL_BRANCH: str = 'if (in.dtype == "bool") {'
_INT32_BRANCH: str = '} else if (in.dtype == "int32") {'
_FP32_BRANCH: str = "} else {"

_WRITE_LOOP_START: str = "for i, t in enumerate(inputs):"
_WRITE_LOOP_END: str = "input_entries.append("

# manifest input dtype -> (cases.py generator, defining module, dtype literal)
_DTYPE_GENERATORS: dict[str, tuple[str, Path, str]] = {
    "bool": ("where_cond_gen", _OPS_DIR / "test_where.py", "torch.bool"),
    "int32": ("to_copy_int_input", _OPS_DIR / "test_to_copy.py", "torch.int32"),
    "float32": ("to_copy_float_input", _OPS_DIR / "test_to_copy.py", "torch.float32"),
}

# The op names the D10 artifact command passes to `generate_op_tests --ops`.
_D10_SUITES: tuple[str, ...] = (
    "split_with_sizes_copy",
    "to_copy",
    "to_copy_f2i",
    "expand_copy",
    "gather",
    "where",
    "compare_scalar",
    "logical_not",
    "index",
    "sub",
)


class TypedInputContractTest(unittest.TestCase):
    def _region(self, text: str, start: str, end: str) -> str:
        """Source between two anchors, each required to occur exactly once."""
        self.assertEqual(text.count(start), 1, f"anchor not unique: {start!r}")
        self.assertEqual(text.count(end), 1, f"anchor not unique: {end!r}")
        begin = text.index(start)
        stop = text.index(end)
        self.assertLess(begin, stop, f"{start!r} must precede {end!r}")
        return text[begin:stop]

    def _input_loop(self) -> str:
        return self._region(_DRIVER.read_text(), _INPUT_LOOP_START, _INPUT_LOOP_END)

    def _function_source(self, path: Path, name: str) -> str:
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                segment = ast.get_source_segment(source, node)
                self.assertIsNotNone(segment, f"no source for {name} in {path}")
                return segment or ""
        self.fail(f"{path} defines no function named {name!r}")

    def test_driver_keeps_all_three_typed_input_branches(self) -> None:
        loop = self._input_loop()
        for branch in (_BOOL_BRANCH, _INT32_BRANCH, _FP32_BRANCH):
            self.assertEqual(loop.count(branch), 1, f"missing branch: {branch!r}")
        self.assertLess(loop.index(_BOOL_BRANCH), loop.index(_INT32_BRANCH))
        self.assertLess(loop.index(_INT32_BRANCH), loop.index(_FP32_BRANCH))

    def test_driver_loads_bool_inputs_as_int8_scalartype_bool(self) -> None:
        branch = self._region(self._input_loop(), _BOOL_BRANCH, _INT32_BRANCH)
        self.assertIn("load_int8_bin(in.path, n)", branch)
        self.assertIn("executorch::aten::ScalarType::Bool", branch)
        # A bool input widened to fp32 (or narrowed to int32) is the mutation.
        self.assertNotIn("load_fp32_bin", branch)
        self.assertNotIn("load_int32_bin", branch)

    def test_driver_loads_int32_inputs_as_int32(self) -> None:
        branch = self._region(self._input_loop(), _INT32_BRANCH, _FP32_BRANCH)
        self.assertIn("load_int32_bin(in.path, n)", branch)
        self.assertNotIn("load_int8_bin", branch)
        self.assertNotIn("load_fp32_bin", branch)

    def test_driver_falls_back_to_fp32_only_for_untyped_inputs(self) -> None:
        loop = self._input_loop()
        branch = loop[loop.index(_FP32_BRANCH) :]
        self.assertIn("load_fp32_bin(in.path, n)", branch)
        self.assertNotIn("ScalarType::Bool", branch)
        self.assertNotIn("load_int8_bin", branch)

    def test_generator_emits_exactly_the_driver_input_dtypes(self) -> None:
        write_loop = self._region(
            _GENERATOR.read_text(), _WRITE_LOOP_START, _WRITE_LOOP_END
        )
        emitted = re.findall(r'in_dtype = "(\w+)"', write_loop)
        self.assertEqual(emitted, ["bool", "int32", "float32"])
        consumed = set(re.findall(r'in\.dtype == "(\w+)"', self._input_loop()))
        # The driver names its two typed branches; fp32 is the unnamed fallback.
        self.assertEqual(consumed | {"float32"}, set(emitted))

    def test_generator_writes_bool_inputs_as_int8_not_fp32(self) -> None:
        write_loop = self._region(
            _GENERATOR.read_text(), _WRITE_LOOP_START, _WRITE_LOOP_END
        )
        branch = self._region(
            write_loop, "if t.dtype == torch.bool:", "elif t.dtype == torch.int32:"
        )
        self.assertIn("_write_int8", branch)
        self.assertNotIn("_write_fp32", branch)

    def test_generator_preserves_int32_golden_width(self) -> None:
        source = self._function_source(_GENERATOR, "_write_golden_output")
        start = source.index("elif raw.dtype == torch.int32:")
        stop = source.index("\n    else:", start)
        branch = source[start:stop]
        self.assertIn('out_dtype = "int32"', branch)
        self.assertNotIn("to(torch.int64)", branch)

    def test_driver_compares_int32_outputs_exactly(self) -> None:
        source = _DRIVER.read_text()
        branch = self._region(
            source,
            '} else if (e_.golden.dtype == "int32") {',
            '} else if (e_.golden.dtype == "int64") {',
        )
        self.assertIn("load_int32_bin", branch)
        self.assertIn("const_data_ptr<int32_t>()", branch)
        self.assertIn("ScalarType::Int", branch)
        self.assertNotIn("within_tol", branch)

    def test_generator_materializes_bool_inputs_unchanged(self) -> None:
        source = self._function_source(_GENERATOR, "_materialize")
        materialize_bool = self._region(
            source, "if _t.dtype == torch.bool:", "return ("
        )
        self.assertIn("return _t", materialize_bool)
        self.assertNotIn("to(torch.int32)", materialize_bool)
        self.assertNotIn("to(torch.float32)", materialize_bool)

    def test_cases_wire_one_input_generator_per_manifest_dtype(self) -> None:
        cases = _CASES.read_text()
        for dtype, (gen, path, literal) in _DTYPE_GENERATORS.items():
            with self.subTest(dtype=dtype):
                self.assertIn(f"gen={gen}", cases, f"no {dtype} input generator wired")
                self.assertIn(literal, self._function_source(path, gen))

    def test_cases_register_every_d10_manifest_suite(self) -> None:
        cases = _CASES.read_text()
        for op in _D10_SUITES:
            with self.subTest(op=op):
                self.assertEqual(cases.count(f'@register_op_test("{op}")'), 1)
