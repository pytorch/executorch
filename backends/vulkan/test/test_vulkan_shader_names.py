# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that generated shader names match the names the dispatcher asks for.

A kernel name is built at runtime rather than looked up: `add_binary_op_node`
concatenates "binary_", the op, a storage suffix and a dtype suffix, then hands
the result to `VK_KERNEL_FROM_STR`. The yaml, meanwhile, is free to name a
variant anything at all. Nothing connects the two, so a variant whose name does
not follow that shape compiles a shader nothing references and leaves the name
the dispatcher wants missing.

That failure is invisible until dispatch. The op is still registered as
supported, so the partitioner claims it, the export succeeds, and the model
aborts on device with "Could not find ShaderInfo with name ...". These tests
close the gap at codegen time instead.

The generator is loaded by file path, and the suffixes are read out of the C++
that produces them, so this needs neither a built runtime nor a GPU and does not
drift when a dtype is added.
"""

import importlib.util
import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VULKAN_ROOT = _REPO_ROOT / "backends" / "vulkan"
_GLSL_DIR = _VULKAN_ROOT / "runtime" / "graph" / "ops" / "glsl"
_SHADER_NAME_UTILS = (
    _VULKAN_ROOT / "runtime" / "graph" / "ops" / "utils" / "ShaderNameUtils.cpp"
)

_spec = importlib.util.spec_from_file_location(
    "gen_vulkan_spv", _VULKAN_ROOT / "runtime" / "gen_vulkan_spv.py"
)
gen_vulkan_spv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_vulkan_spv)

# The yaml templates whose variants `add_binary_op_node` dispatches into.
_BINARY_TEMPLATES = ("binary_op_buffer", "binary_op_texture")


def _function_body(text: str, name: str) -> str:
    """Source of a top-level C++ function, from its signature to its closing brace."""
    start = next(
        (
            i
            for i, line in enumerate(text.splitlines())
            if re.match(rf"^\w[\w:<>&* ]*\b{re.escape(name)}\(", line)
        ),
        None,
    )
    if start is None:
        raise AssertionError(f"{name} not found in {_SHADER_NAME_UTILS}")
    lines = text.splitlines()[start:]
    end = next(i for i, line in enumerate(lines) if line == "}")
    return "\n".join(lines[: end + 1])


def _suffixes(function_name: str) -> tuple:
    """Every literal suffix one ShaderNameUtils.cpp helper can append.

    Read from the C++ rather than restated here: a dtype added to
    `add_dtype_suffix` becomes legal in a shader name the moment it is added,
    and a test carrying its own copy of the list would reject it.
    """
    body = _function_body(_SHADER_NAME_UTILS.read_text(), function_name)
    found = re.findall(r'kernel_name \+= "(_[a-z0-9]+)";', body)
    if not found:
        raise AssertionError(f"no suffixes parsed out of {function_name}")
    return tuple(dict.fromkeys(found))


STORAGE_SUFFIXES = _suffixes("add_storage_type_suffix")
DTYPE_SUFFIXES = _suffixes("add_dtype_suffix")


def _generated_names() -> dict:
    """Variant names the shader codegen produces, keyed by yaml template."""
    env = dict(gen_vulkan_spv.DEFAULT_ENV)
    env.update(gen_vulkan_spv.TYPE_MAPPINGS)
    env.update(gen_vulkan_spv.UTILITY_FNS)
    generator = gen_vulkan_spv.SPVGenerator([str(_GLSL_DIR)], env, glslc_path=None)
    return {
        template: [variant["NAME"] for variant in variants]
        for template, variants in generator.shader_template_params.items()
    }


class TestShaderNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.names = _generated_names()

    def test_suffixes_are_parsed_from_the_cpp(self) -> None:
        # Guards the two tests below: a parse that silently returned something
        # empty or wrong would make them pass by vacuously accepting any name.
        self.assertIn("_buffer", STORAGE_SUFFIXES)
        self.assertIn("_texture3d", STORAGE_SUFFIXES)
        self.assertIn("_float", DTYPE_SUFFIXES)
        self.assertIn("_int32", DTYPE_SUFFIXES)

    def test_int32_eq_shaders_are_named_for_the_dispatcher(self) -> None:
        """An int32 `aten.eq.Tensor` must find a shader on both storage types.

        Declared as `binary_eq_int32_{buffer,texture3d}` for a while, which put
        the dtype in the middle and so generated a name no dispatch could ever
        build. Kokoro's synthesizer aborted on it.
        """
        for template, expected in (
            ("binary_op_buffer", "binary_eq_buffer_int32"),
            ("binary_op_texture", "binary_eq_texture3d_int32"),
        ):
            with self.subTest(template=template):
                self.assertIn(expected, self.names[template])

    def test_binary_variants_end_in_a_storage_and_dtype_suffix(self) -> None:
        """The general form of the same bug, for every binary op at once.

        `add_binary_op_node` appends the storage suffix and then the dtype
        suffix, in that order, to every name it builds. A generated variant that
        does not end that way cannot be reached from the dispatcher, whatever
        else is true about it.
        """
        legal = tuple(
            storage + dtype for storage in STORAGE_SUFFIXES for dtype in DTYPE_SUFFIXES
        )
        unreachable = [
            name
            for template in _BINARY_TEMPLATES
            for name in self.names[template]
            if not name.endswith(legal)
        ]
        self.assertEqual(
            unreachable,
            [],
            "these shaders are generated but no dispatch can name them; "
            "see add_binary_op_node in BinaryOp.cpp",
        )


if __name__ == "__main__":
    unittest.main()
