# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import re
import unittest

WASM_DIR = pathlib.Path(__file__).parents[1]
EXECUTORCH_ROOT = WASM_DIR.parents[1]
OPTION = "EXECUTORCH_BUILD_WASM_WEBGPU"
# Emscripten 4.0.13 is the first release whose emdawnwebgpu port provides the
# instance-features API (WGPUInstanceFeatureName, requiredFeatures) that
# WebGPUDevice.cpp compiles against outside the vendored-Dawn branch.
MIN_EMSCRIPTEN_VERSION = (4, 0, 13)


def _guarded_blocks(source: str, condition: str) -> list[str]:
    """Bodies of every top-level `if(<condition>)` block in a CMake file."""
    lines = source.splitlines()
    command_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    condition_pattern = re.compile(
        rf"^\s*if\s*\(\s*{re.escape(condition)}\s*\)\s*$", re.IGNORECASE
    )
    blocks = []
    start = None
    depth = 0
    for index, line in enumerate(lines):
        if start is None:
            if condition_pattern.match(line):
                start = index
                depth = 1
            continue
        match = command_pattern.match(line)
        if match is None:
            continue
        command = match.group(1).lower()
        if command == "if":
            depth += 1
        elif command == "endif":
            depth -= 1
            if depth == 0:
                blocks.append("\n".join(lines[start + 1 : index]))
                start = None
    if start is not None:
        raise AssertionError(f"if({condition}) has no matching endif()")
    if not blocks:
        raise AssertionError(f"if({condition}) block not found")
    return blocks


def _read(*parts: str) -> str:
    return (EXECUTORCH_ROOT.joinpath(*parts)).read_text()


def _read_root(*parts: str) -> str:
    """Read a repo-root file from the top level or from oss/."""
    for root in (EXECUTORCH_ROOT, EXECUTORCH_ROOT / "oss"):
        path = root.joinpath(*parts)
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(EXECUTORCH_ROOT.joinpath(*parts))


class TestWebGPUWasmOption(unittest.TestCase):
    def test_block_parser_finds_every_block_and_skips_nested_endif(self) -> None:
        source = """
if(OPT)
  if(INNER)
    inner_command()
  endif()
  first_command()
endif()
other_command()
if(OPT)
  second_command()
endif()
"""
        blocks = _guarded_blocks(source, "OPT")

        self.assertEqual(len(blocks), 2)
        self.assertIn("inner_command()", blocks[0])
        self.assertIn("first_command()", blocks[0])
        self.assertNotIn("other_command()", blocks[0])
        self.assertIn("second_command()", blocks[1])

    def test_option_defaults_off_and_requires_wasm_and_webgpu(self) -> None:
        preset = _read("tools", "cmake", "preset", "default.cmake")

        self.assertRegex(
            preset,
            rf"define_overridable_option\(\s*{OPTION}\s+\"[^\"]+\"\s+BOOL\s+OFF\s*\)",
        )
        self.assertRegex(
            preset,
            rf"check_required_options_on\(\s*IF_ON\s+{OPTION}\s+REQUIRES\s+"
            r"EXECUTORCH_BUILD_WASM\s+EXECUTORCH_BUILD_WEBGPU\s*\)",
        )

    def test_backend_is_added_before_the_wasm_extension(self) -> None:
        root = _read_root("CMakeLists.txt")
        webgpu = root.index(
            "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/webgpu)"
        )
        wasm = root.index(
            "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/wasm)"
        )

        self.assertLess(webgpu, wasm)

    def test_webgpu_build_runs_configuration_contract_tests(self) -> None:
        script = _read("scripts", "build_wasm_tests.sh")

        self.assertIn(
            "python3 -m unittest discover \\\n      -s extension/wasm/test -p test_webgpu_configuration.py",
            script,
        )


class TestWebGPUWasmCMake(unittest.TestCase):
    def setUp(self) -> None:
        self.cmake = (WASM_DIR / "CMakeLists.txt").read_text()
        self.blocks = _guarded_blocks(self.cmake, OPTION)
        self.guarded = "\n".join(self.blocks)

    def test_backend_is_linked_only_when_the_option_is_on(self) -> None:
        self.assertIn("list(APPEND link_libraries webgpu_backend)", self.guarded)

        outside = self.cmake
        for block in self.blocks:
            outside = outside.replace(block, "")
        self.assertNotIn("webgpu_backend", outside)
        self.assertNotIn("ASYNCIFY", outside)
        self.assertNotIn("ALLOW_MEMORY_GROWTH", outside)
        self.assertNotIn("MAXIMUM_MEMORY", outside)
        self.assertNotIn("fexceptions", outside)

    def test_missing_backend_target_fails_configuration(self) -> None:
        self.assertIn("if(NOT TARGET webgpu_backend)", self.guarded)
        self.assertIn("FATAL_ERROR", self.guarded)

    def test_asyncify_is_enabled_without_ignoring_indirect_calls(self) -> None:
        self.assertIn("-sASYNCIFY", self.guarded)
        directives = "\n".join(
            line
            for line in self.cmake.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn("ASYNCIFY_IGNORE_INDIRECT", directives)

    def test_heap_can_grow_to_the_wasm32_limit(self) -> None:
        self.assertIn("-sALLOW_MEMORY_GROWTH=1", self.guarded)
        self.assertIn("-sMAXIMUM_MEMORY=4GB", self.guarded)

    def test_exceptions_are_enabled_for_compile_and_link(self) -> None:
        self.assertRegex(
            self.guarded,
            r"target_compile_options\(\s*executorch_wasm\s+PUBLIC\s+-fexceptions\s*\)",
        )
        self.assertRegex(
            self.guarded,
            r"target_link_options\(\s*executorch_wasm\s+PUBLIC\s+-fexceptions\s*\)",
        )

    def test_bindings_are_told_the_backend_is_present(self) -> None:
        self.assertRegex(
            self.guarded,
            r"target_compile_definitions\(\s*executorch_wasm\s+PUBLIC\s+"
            r"EXECUTORCH_WASM_WEBGPU\s*\)",
        )


class TestWebGPUWasmBindings(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (WASM_DIR / "wasm_bindings.cpp").read_text()

    def _webgpu_sections(self) -> str:
        return "\n".join(
            re.findall(
                r"#ifdef EXECUTORCH_WASM_WEBGPU\n(.*?)#endif",
                self.source,
                re.DOTALL,
            )
        )

    def test_every_webgpu_symbol_is_behind_the_compile_definition(self) -> None:
        sections = self._webgpu_sections()

        for symbol in (
            "WebGPUDevice.h",
            "create_webgpu_context",
            "destroy_webgpu_context",
            "set_default_webgpu_context",
            'function("initWebGPU", &init_webgpu)',
        ):
            self.assertIn(symbol, sections)
            self.assertEqual(self.source.count(symbol), sections.count(symbol))

    def test_context_is_created_once_and_registered(self) -> None:
        initialize = re.search(
            r"void initialize\(\) \{(.*?)\n  \}", self.source, re.DOTALL
        )
        self.assertIsNotNone(initialize)
        body = initialize.group(1)

        self.assertIn("if (context_) {\n      return;\n    }", body)
        self.assertLess(
            body.index("set_default_webgpu_context(context.get())"),
            body.index("context_ = std::move(context)"),
        )

    def test_context_is_unregistered_before_it_is_destroyed(self) -> None:
        destructor = re.search(
            r"~WebGPUContextOwner\(\) \{(.*?)\n  \}", self.source, re.DOTALL
        )
        self.assertIsNotNone(destructor)
        body = destructor.group(1)

        self.assertLess(
            body.index("set_default_webgpu_context(nullptr)"),
            body.index("destroy_webgpu_context(*context_)"),
        )

    def test_initialization_failure_becomes_a_js_error(self) -> None:
        self.assertRegex(
            self.source,
            r"catch \(const std::exception& e\) \{\s*"
            r"THROW_JS_ERROR\(Error, \"Failed to initialize WebGPU: %s\", e\.what\(\)\);",
        )

    def test_no_jspi_only_async_policy(self) -> None:
        # embind's async() policy asserts under -sASYNCIFY; it needs JSPI.
        self.assertNotIn("emscripten::async", self.source)
        self.assertNotIn("async()", self.source)


class TestWebGPUWasmScripts(unittest.TestCase):
    def test_build_script_builds_the_webgpu_configuration_separately(self) -> None:
        script = _read("scripts", "build_wasm_tests.sh")

        self.assertIn('"$arg" == "--webgpu"', script)
        self.assertIn("-DEXECUTORCH_BUILD_WEBGPU=ON", script)
        self.assertIn(f"-D{OPTION}=ON", script)
        self.assertIn("${ETDUMP_OPTS} ${WEBGPU_OPTS}", script)
        self.assertIn("CMAKE_OUT=cmake-out-wasm-webgpu", script)

    def test_ci_pins_an_emscripten_with_the_instance_features_api(self) -> None:
        script = _read_root(".ci", "scripts", "setup-emscripten.sh")
        match = re.search(r'EMSCRIPTEN_VERSION:-(\d+)\.(\d+)\.(\d+)\}"', script)
        self.assertIsNotNone(match)
        version = tuple(int(part) for part in match.groups())

        self.assertGreaterEqual(version, MIN_EMSCRIPTEN_VERSION)
        self.assertIn('./emsdk install "${emscripten_version}"', script)
        self.assertIn('./emsdk activate "${emscripten_version}"', script)

        readme = (WASM_DIR / "README.md").read_text()
        pinned = ".".join(str(part) for part in version)
        self.assertIn(f"./emsdk install {pinned}", readme)
        self.assertIn(f"./emsdk activate {pinned}", readme)

    def test_ci_builds_the_webgpu_configuration(self) -> None:
        workflow = _read_root(".github", "workflows", "pull.yml")

        self.assertIn("bash scripts/build_wasm_tests.sh --webgpu", workflow)


if __name__ == "__main__":
    unittest.main()
