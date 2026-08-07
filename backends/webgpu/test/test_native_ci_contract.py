# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import ast
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

_BACKEND: pathlib.Path = pathlib.Path(__file__).parents[1]
_EXECUTORCH: pathlib.Path = _BACKEND.parents[1]
_GEMMA4_TESTS: pathlib.Path = _EXECUTORCH / "examples/models/gemma4/tests"

_CI_SCRIPT: pathlib.Path = _BACKEND / "scripts/test_webgpu_native_ci.sh"
_CMAKE: pathlib.Path = _BACKEND / "CMakeLists.txt"
_DYNAMIC_SHAPE_TEST: pathlib.Path = _BACKEND / "test/native/test_dynamic_shape.cpp"
_UPDATE_CACHE_TEST: pathlib.Path = _BACKEND / "test/native/test_update_cache.cpp"
_SLICE_IMPL: pathlib.Path = _BACKEND / "runtime/ops/slice/Slice.cpp"
_SLICE_DISPATCH: pathlib.Path = _BACKEND / "runtime/ops/slice/SliceDispatch.h"

_THIS_GATE = "backends/webgpu/test/test_native_ci_contract.py"

# Directories D10 adds or edits test sources in; the drift guard narrows source
# control to these so an unrelated working-copy edit cannot redden it.
_D10_TEST_SURFACE: tuple[str, ...] = (
    "backends/webgpu/test/",
    "examples/models/gemma4/tests/",
)

# Every test source D10 adds or edits, executorch-relative.
_D10_TEST_SOURCES: tuple[str, ...] = (
    "backends/webgpu/test/native/test_q4gsw_m3.cpp",
    "backends/webgpu/test/native/test_scatter.cpp",
    "backends/webgpu/test/native/test_topk.cpp",
    "backends/webgpu/test/op_tests/test_typed_input_contract.py",
    "backends/webgpu/test/ops/index/test_index.py",
    "backends/webgpu/test/ops/scatter/test_scatter.py",
    "backends/webgpu/test/ops/test_gather.py",
    "backends/webgpu/test/ops/test_to_copy.py",
    "backends/webgpu/test/ops/test_where.py",
    "backends/webgpu/test/ops/topk/test_topk.py",
    "backends/webgpu/test/test_native_ci_contract.py",
    "examples/models/gemma4/tests/test_eagle_combined_round.py",
    "examples/models/gemma4/tests/test_export_assistant_webgpu_artifacts.py",
    "examples/models/gemma4/tests/test_export_partitioners.py",
    "examples/models/gemma4/tests/test_gemma4_spec_runner_contract.cpp",
    "examples/models/gemma4/tests/test_mtp_spec_oracle.py",
    "examples/models/gemma4/tests/test_oss_source_closure.py",
    "examples/models/gemma4/tests/test_webgpu_artifact_manifest.py",
    "examples/models/gemma4/tests/test_webgpu_spec_contract.py",
)

# Build files whose registrations must resolve to real sources on disk.
_BUILD_FILES: tuple[pathlib.Path, ...] = (
    _BACKEND / "test/BUCK",
    _BACKEND / "test/targets.bzl",
    _GEMMA4_TESTS / "targets.bzl",
)

# Buck package -> every build file allowed to define that package's targets.
_PACKAGE_BUILD_FILES: dict[str, tuple[pathlib.Path, ...]] = {
    "//backends/webgpu/test": (
        _BACKEND / "test/BUCK",
        _BACKEND / "test/targets.bzl",
    ),
    "//examples/models/gemma4/tests": (_GEMMA4_TESTS / "targets.bzl",),
}

# GEMMA4_D10_MTP_BUCK_TEST_TARGETS from the D10 command contract, verbatim.
_MTP_BUCK_TEST_TARGETS: tuple[str, ...] = (
    "//examples/models/gemma4/tests:test_eagle_combined_round",
    "//examples/models/gemma4/tests:test_export_assistant_webgpu_artifacts",
    "//examples/models/gemma4/tests:test_export_partitioners",
    "//examples/models/gemma4/tests:test_webgpu_artifact_manifest",
    "//examples/models/gemma4/tests:test_mtp_spec_oracle",
    "//examples/models/gemma4/tests:test_webgpu_spec_contract",
    "//backends/webgpu/test:test_scatter_cpu",
    "//backends/webgpu/test:test_topk_cpu",
    "//backends/webgpu/test:test_to_copy",
    "//backends/webgpu/test:test_index",
)

# GEMMA4_D10_PLAIN_REGRESSION_TARGETS from the same contract, verbatim.
_PLAIN_REGRESSION_TARGETS: tuple[str, ...] = (
    "//examples/models/gemma4/tests:test_webgpu_artifact_manifest",
    "//examples/models/gemma4/tests:test_export_partitioners",
    "//examples/models/gemma4/tests:test_export_smoke",
    "//examples/models/gemma4/tests:test_selected_row_cross_decoder",
    "//backends/webgpu/test:test_et_vk_sdpa",
    "//backends/webgpu/test:test_rope_hf_single",
)

# Executables D10's documented `cmake --build --target ...` line names.
_CMAKE_BUILD_TARGETS: tuple[str, ...] = (
    "webgpu_native_test",
    "webgpu_dynamic_shape_test",
    "webgpu_update_cache_test",
    "webgpu_op_test",
    "webgpu_scatter_test",
    "webgpu_topk_test",
)


def _bash_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(name)}=\((.*?)\)", source, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise AssertionError(f"{name} Bash array not found")
    return shlex.split(match.group(1))


def _bash_function(source: str, name: str) -> str:
    pattern = "^" + re.escape(name) + r"\(\) \{\n(.*?)^\}"
    match = re.search(
        pattern,
        source,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{name} Bash function not found")
    return match.group(1)


def _run_required_gtests(
    script: str, output: str, status: int = 0
) -> subprocess.CompletedProcess[str]:
    program = f"""
run_with_required_device() {{
{_bash_function(script, "run_with_required_device")}
}}
run_required_gtests() {{
{_bash_function(script, "run_required_gtests")}
}}
fake_gtest() {{
  printf '%s\\n' "$1"
  return "$2"
}}
run_required_gtests fake_gtest "$1" "$2"
"""
    return subprocess.run(
        ["bash", "-c", program, "required-gtests", output, str(status)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _run_recreate_exact_directory(
    script: str, target: pathlib.Path, expected: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    program = f"""
recreate_exact_directory() {{
{_bash_function(script, "recreate_exact_directory")}
}}
recreate_exact_directory "$1" "$2"
"""
    return subprocess.run(
        ["bash", "-c", program, "recreate-exact-directory", str(target), str(expected)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _ci_script() -> str:
    return _CI_SCRIPT.read_text()


def _required_unique_binding(source: str, name: str, value: str, before: int) -> None:
    pattern = re.compile(
        rf"^[ \t]*(?:(?:export|readonly|declare|typeset)"
        rf"(?:[ \t]+-[A-Za-z]+)?[ \t]+)?{re.escape(name)}=[^\n]*$",
        re.MULTILINE,
    )
    bindings = list(pattern.finditer(source))
    expected = f"{name}={value}"
    if len(bindings) != 1 or bindings[0].group() != expected:
        raise AssertionError(
            f"expected one canonical {name} binding, got "
            f"{[binding.group() for binding in bindings]}"
        )
    if bindings[0].start() >= before:
        raise AssertionError(f"{name} must be bound before required Slice tests")


def _required_gtest_invocation(source: str) -> tuple[str, ...]:
    starts = list(
        re.finditer(r"^[ \t]*run_required_gtests[ \t]+", source, re.MULTILINE)
    )
    if len(starts) != 1:
        raise AssertionError(
            f"expected one run_required_gtests invocation, got {len(starts)}"
        )
    _required_unique_binding(
        source, "DYNAMIC_SHAPE_DIR", '"/tmp/dynamic_shape"', starts[0].start()
    )
    _required_unique_binding(
        source,
        "BIN_DIR",
        '"${BUILD_DIR}/backends/webgpu"',
        starts[0].start(),
    )

    continued = re.sub(r"\\\n[ \t]*", " ", source)
    commands = [
        line.lstrip()
        for line in continued.splitlines()
        if line.lstrip().startswith("run_required_gtests ")
    ]
    if len(commands) != 1:
        raise AssertionError(
            f"expected one run_required_gtests invocation, got {len(commands)}"
        )
    program = f"""
run_required_gtests() {{
  printf '%s\\n' "$@"
}}
BIN_DIR="$1"
DYNAMIC_SHAPE_DIR="$2"
{commands[0]}
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            program,
            "required-gtest-invocation",
            "/contract bin",
            "/contract dynamic shape",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return tuple(completed.stdout.splitlines())


def _required_slice_contract_result(script: str) -> unittest.TestResult:
    global _CI_SCRIPT
    with tempfile.TemporaryDirectory() as temporary:
        mutated_script = pathlib.Path(temporary) / "test_webgpu_native_ci.sh"
        mutated_script.write_text(script)
        original_script = _CI_SCRIPT
        try:
            _CI_SCRIPT = mutated_script
            result = unittest.TestResult()
            TestNativeCIContract(
                "test_runs_required_slice_regressions_fail_closed"
            ).run(result)
            return result
        finally:
            _CI_SCRIPT = original_script


def _srcs_entries(source: str) -> list[str]:
    entries: list[str] = []
    for block in re.findall(r"srcs\s*=\s*\[(.*?)\]", source, re.DOTALL):
        entries.extend(re.findall(r'"([^"]+)"', block))
    return entries


def _target_names(source: str) -> set[str]:
    return set(re.findall(r'name\s*=\s*"([^"]+)"', source))


def _cmake_defines(cmake: str, name: str) -> bool:
    """A bare substring match lets a longer target name mask a deleted one."""
    pattern = rf"(?:add_webgpu_native_test|add_executable)\(\s*{re.escape(name)}\b"
    return re.search(pattern, cmake) is not None


def _undefined_labels(labels: tuple[str, ...]) -> list[str]:
    undefined: list[str] = []
    for label in labels:
        package, _, name = label.partition(":")
        defined: set[str] = set()
        for build_file in _PACKAGE_BUILD_FILES[package]:
            defined |= _target_names(build_file.read_text())
        if name not in defined:
            undefined.append(label)
    return undefined


def _sl_status_paths() -> list[str] | None:
    """Executorch-relative added/modified paths, or None when `sl` cannot answer."""
    try:
        completed = subprocess.run(
            ["sl", "status", "--reason", "D10 registration contract"],
            capture_output=True,
            cwd=_EXECUTORCH,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return [
        line[2:]
        for line in completed.stdout.splitlines()
        if line[:2] in ("M ", "A ") and not line[2:].startswith("..")
    ]


def _sl_log_node(relative: str) -> str:
    """Node that last touched a path; empty while the path is uncommitted."""
    args = ["sl", "log", relative, "-T", "{node}\n", "-l", "1"]
    try:
        completed = subprocess.run(
            args + ["--reason", "D10 registration contract"],
            capture_output=True,
            cwd=_EXECUTORCH,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    lines = completed.stdout.splitlines()
    return lines[0] if lines else ""


def _sapling_is_usable() -> bool:
    """True when `sl` and a Sapling working copy are both present."""
    if shutil.which("sl") is None:
        return False
    return any(
        (parent / ".sl").is_dir() or (parent / ".hg").is_dir()
        for parent in (_EXECUTORCH, *_EXECUTORCH.parents)
    )


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _defines_test_case(path: pathlib.Path) -> bool:
    source = path.read_text()
    if path.suffix == ".cpp":
        return "TEST(" in source or "TEST_F(" in source
    return any(
        isinstance(node, ast.ClassDef)
        and any(_base_name(base).endswith("TestCase") for base in node.bases)
        for node in ast.walk(ast.parse(source))
    )


def _is_test_source(relative: str) -> bool:
    path = pathlib.PurePosixPath(relative)
    if path.suffix not in (".py", ".cpp"):
        return False
    return any(part in ("test", "tests") for part in path.parts[:-1])


def _reported_test_sources() -> list[str] | None:
    """Test sources source control reports inside D10's surface, or None."""
    reported = _sl_status_paths()
    if reported is None:
        return None
    return sorted(
        relative
        for relative in reported
        if relative.startswith(_D10_TEST_SURFACE)
        and _is_test_source(relative)
        and (_EXECUTORCH / relative).is_file()
        and _defines_test_case(_EXECUTORCH / relative)
    )


def _invocation_text() -> tuple[str, list[str]]:
    """The CI script's commands, comments dropped and continuations joined."""
    body = "\n".join(
        line for line in _ci_script().splitlines() if not line.lstrip().startswith("#")
    )
    joined = re.sub(r"\\\n\s*", " ", body)
    return joined, re.findall(r'-c\s+"(.*?)"', joined, re.DOTALL)


def _resolve_module(dotted: str) -> str | None:
    parts = dotted.split(".")[1:]
    while parts:
        candidate = pathlib.PurePosixPath(*parts).with_suffix(".py")
        if (_EXECUTORCH / candidate).is_file():
            return str(candidate)
        parts.pop()
    return None


def _ci_script_modules() -> set[str]:
    """Executorch-relative sources the CI script actually invokes, not mentions."""
    joined, programs = _invocation_text()
    dotted_names = re.findall(
        r"-m\s+(?:unittest\s+)?(executorch(?:\.[A-Za-z_]\w*)+)", joined
    )
    for program in programs:
        dotted_names += re.findall(r"executorch(?:\.[A-Za-z_]\w*)+", program)
    invoked: set[str] = set()
    for dotted in dotted_names:
        resolved = _resolve_module(dotted)
        if resolved is not None:
            invoked.add(resolved)
    return invoked


def _registered_sources() -> set[str]:
    registered: set[str] = set()
    for build_file in _BUILD_FILES:
        for entry in _srcs_entries(build_file.read_text()):
            resolved = (build_file.parent / entry).resolve()
            registered.add(str(resolved.relative_to(_EXECUTORCH.resolve())))
    for entry in re.findall(r"[\w][\w./]*\.cpp", _CMAKE.read_text()):
        registered.add(str(pathlib.PurePosixPath("backends/webgpu") / entry))
    return registered | _ci_script_modules()


class TestNativeCIContract(unittest.TestCase):
    def test_required_slice_contract_rejects_missing_heavy_env(self) -> None:
        script = _ci_script()
        invocation = (
            "run_required_gtests env WEBGPU_REQUIRE_DEVICE=1 "
            "WEBGPU_TEST_HEAVY=1 \\\n"
        )
        self.assertEqual(1, script.count(invocation))
        result = _required_slice_contract_result(
            script.replace(
                invocation,
                "run_required_gtests env WEBGPU_REQUIRE_DEVICE=1 \\\n",
            )
        )

        self.assertFalse(result.wasSuccessful())
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.failures))

    def test_required_slice_contract_rejects_stale_artifact_directory(self) -> None:
        script = _ci_script()
        invocation = (
            '"${BIN_DIR}/webgpu_dynamic_shape_test" "${DYNAMIC_SHAPE_DIR}" \\\n'
        )
        self.assertEqual(1, script.count(invocation))
        result = _required_slice_contract_result(
            script.replace(
                invocation,
                '"${BIN_DIR}/webgpu_dynamic_shape_test" '
                '"/tmp/stale_dynamic_shape" \\\n',
            )
        )

        self.assertFalse(result.wasSuccessful())
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.failures))

    def test_required_slice_contract_rejects_late_artifact_rebinding(self) -> None:
        script = _ci_script()
        invocation = (
            "run_required_gtests env WEBGPU_REQUIRE_DEVICE=1 "
            "WEBGPU_TEST_HEAVY=1 \\\n"
        )
        self.assertEqual(1, script.count(invocation))
        result = _required_slice_contract_result(
            script.replace(
                invocation,
                "DYNAMIC_SHAPE_DIR=/tmp/stale_dynamic_shape\n" + invocation,
            )
        )

        self.assertFalse(result.wasSuccessful())
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.failures))

    def test_required_slice_contract_rejects_literal_variable_arguments(self) -> None:
        script = _ci_script()
        invocation = (
            '"${BIN_DIR}/webgpu_dynamic_shape_test" "${DYNAMIC_SHAPE_DIR}" \\\n'
        )
        self.assertEqual(1, script.count(invocation))
        result = _required_slice_contract_result(
            script.replace(
                invocation,
                "'${BIN_DIR}/webgpu_dynamic_shape_test' " "'${DYNAMIC_SHAPE_DIR}' \\\n",
            )
        )

        self.assertFalse(result.wasSuccessful())
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.failures))

    def test_required_slice_contract_accepts_leading_indentation(self) -> None:
        script = _ci_script()
        invocation = (
            "run_required_gtests env WEBGPU_REQUIRE_DEVICE=1 "
            "WEBGPU_TEST_HEAVY=1 \\\n"
        )
        self.assertEqual(1, script.count(invocation))
        result = _required_slice_contract_result(
            script.replace(invocation, "  " + invocation)
        )

        self.assertTrue(result.wasSuccessful())
        self.assertEqual([], result.errors)
        self.assertEqual([], result.failures)

    def test_dynamic_slice_export_is_fresh_heavy_and_fail_closed(self) -> None:
        script = _ci_script()
        recreate = (
            'recreate_exact_directory "${DYNAMIC_SHAPE_DIR}" ' '"/tmp/dynamic_shape"'
        )
        heavy_export = (
            'WEBGPU_TEST_HEAVY=1 $PYTHON_EXECUTABLE -c "\n'
            "from executorch.backends.webgpu.test.ops.dynamic_shape."
            "test_dynamic_shape_export import export_dynamic_shape_cases"
        )
        next_export = (
            '$PYTHON_EXECUTABLE -c "\n'
            "from executorch.backends.webgpu.test.ops.test_sdpa import ("
        )
        closure = script[script.index(recreate) : script.index(next_export)]

        self.assertIn(heavy_export, closure)
        self.assertLess(closure.index(recreate), closure.index(heavy_export))
        for fixture in (
            "${DYNAMIC_SHAPE_DIR}/dyn_slice_2d.pte",
            "${DYNAMIC_SHAPE_DIR}/slice_dual_store.pte",
            "${DYNAMIC_SHAPE_DIR}/slice_dual_store.input.bin",
            "${DYNAMIC_SHAPE_DIR}/slice_dual_store.out0.golden.bin",
            "${DYNAMIC_SHAPE_DIR}/slice_dual_store.out1.golden.bin",
        ):
            requirement = f'require_file "{fixture}"'
            self.assertEqual(script.count(requirement), 1, fixture)
            self.assertIn(requirement, closure)
            self.assertLess(closure.index(heavy_export), closure.index(requirement))

    def test_dynamic_slice_export_cannot_reuse_stale_directory(self) -> None:
        script = _ci_script()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "dynamic_shape"
            target.mkdir()
            stale = target / "dyn_slice_2d.pte"
            stale.write_text("stale")

            recreated = _run_recreate_exact_directory(script, target, target)
            self.assertEqual(0, recreated.returncode, recreated.stderr)
            self.assertTrue(target.is_dir())
            self.assertFalse(stale.exists())

            protected = root / "protected"
            protected.mkdir()
            sentinel = protected / "sentinel"
            sentinel.write_text("keep")
            rejected = _run_recreate_exact_directory(script, protected, target)
            self.assertNotEqual(0, rejected.returncode)
            self.assertTrue(sentinel.is_file())

    def test_runs_required_slice_regressions_fail_closed(self) -> None:
        script = _ci_script()
        helper = _bash_function(script, "run_required_gtests")
        filter_value = (
            "DynamicShape.SliceCrosses2dDispatchBoundary:"
            "DynamicShape.CatCrosses2dDispatchBoundary:"
            "DynamicShape.SliceDualStoreWritesBothDestinations"
        )

        self.assertIn('run_with_required_device "$@"', helper)
        self.assertIn("DynamicShape.SliceCrosses2dDispatchBoundary", helper)
        self.assertIn("DynamicShape.CatCrosses2dDispatchBoundary", helper)
        self.assertIn("DynamicShape.SliceDualStoreWritesBothDestinations", helper)
        self.assertIn("[  PASSED  ] 3 tests.", helper)
        self.assertIn("grep -Eq '^\\[  SKIPPED \\]'", helper)
        self.assertEqual(
            (
                "env",
                "WEBGPU_REQUIRE_DEVICE=1",
                "WEBGPU_TEST_HEAVY=1",
                "/contract bin/webgpu_dynamic_shape_test",
                "/contract dynamic shape",
                f"--gtest_filter={filter_value}",
            ),
            _required_gtest_invocation(script),
        )
        self.assertLess(
            script.index(
                '"${BIN_DIR}/webgpu_dynamic_shape_test" "${DYNAMIC_SHAPE_DIR}"'
            ),
            script.index("run_required_gtests env WEBGPU_REQUIRE_DEVICE=1"),
        )

    def test_update_cache_dynamic_contract_is_fail_closed(self) -> None:
        script = _ci_script()
        recreate = 'recreate_exact_directory "${UPDATE_CACHE_DIR}" "/tmp/update_cache"'
        export_dynamic = (
            "export_dynamic_update_cache('${UPDATE_CACHE_DIR}/dynamic.pte')"
        )
        export_intermediate = (
            "export_intermediate_dynamic_update_cache("
            "'${UPDATE_CACHE_DIR}/dynamic_intermediate.pte')"
        )
        state_run = '"${BIN_DIR}/webgpu_update_cache_state_test"'
        required_run = (
            "run_with_required_device env WEBGPU_REQUIRE_DEVICE=1 \\\n"
            '    WEBGPU_UPDATE_CACHE_DIR="${UPDATE_CACHE_DIR}" \\\n'
            '    "${BIN_DIR}/webgpu_update_cache_test" "${UPDATE_CACHE_DIR}"'
        )

        for fragment in (
            recreate,
            export_dynamic,
            export_intermediate,
            'require_file "${UPDATE_CACHE_DIR}/dynamic.pte"',
            'require_file "${UPDATE_CACHE_DIR}/dynamic_intermediate.pte"',
            state_run,
            required_run,
        ):
            self.assertEqual(script.count(fragment), 1, fragment)
        self.assertLess(script.index(recreate), script.index(export_dynamic))
        self.assertLess(script.index(export_dynamic), script.index(state_run))
        self.assertLess(script.index(state_run), script.index(required_run))

        source = _UPDATE_CACHE_TEST.read_text()
        self.assertIn("test/native/RequiredDevicePolicy.h>", source)
        self.assertIn('std::getenv("WEBGPU_REQUIRE_DEVICE")', source)
        self.assertIn("required_device_failure_exit_code", source)
        self.assertIn('std::printf("WebGPU device acquired (native)\\n")', source)

    def test_required_gtest_helper_rejects_incomplete_output(self) -> None:
        script = _ci_script()
        first = "DynamicShape.SliceCrosses2dDispatchBoundary"
        second = "DynamicShape.CatCrosses2dDispatchBoundary"
        third = "DynamicShape.SliceDualStoreWritesBothDestinations"
        accepted = "\n".join(
            (
                "WebGPU device acquired (native)",
                f"[       OK ] {first} (1 ms)",
                f"[       OK ] {second} (2 ms)",
                f"[       OK ] {third} (3 ms)",
                "[  PASSED  ] 3 tests.",
            )
        )

        passed = _run_required_gtests(script, accepted)
        self.assertEqual(0, passed.returncode, passed.stderr)
        rejected = (
            (accepted.replace("WebGPU device acquired (native)\n", ""), 0),
            (accepted.replace(f"[       OK ] {second} (2 ms)\n", ""), 0),
            (accepted.replace(f"[       OK ] {third} (3 ms)\n", ""), 0),
            (accepted.replace("[  PASSED  ] 3 tests.", "[  PASSED  ] 2 tests."), 0),
            (accepted + "\n[  SKIPPED ] DynamicShape.Unexpected (0 ms)", 0),
            (accepted, 3),
        )
        for output, status in rejected:
            with self.subTest(output=output, status=status):
                result = _run_required_gtests(script, output, status)
                self.assertNotEqual(0, result.returncode, result.stdout)

    def test_dynamic_slice_main_fails_closed_when_device_is_required(self) -> None:
        source = _DYNAMIC_SHAPE_TEST.read_text()

        self.assertIn(
            "test/native/RequiredDevicePolicy.h>",
            source,
        )
        self.assertIn('std::getenv("WEBGPU_REQUIRE_DEVICE")', source)
        self.assertIn("required_device_failure_exit_code", source)
        self.assertIn('std::printf("WebGPU device acquired (native)\\n")', source)

    def test_slice_dispatch_grid_helper_owns_both_dimensions(self) -> None:
        self.assertTrue(_SLICE_DISPATCH.is_file(), _SLICE_DISPATCH)
        header = _SLICE_DISPATCH.read_text()
        implementation = _SLICE_IMPL.read_text()

        self.assertIn("dispatch.workgroup_count_x = grid.x;", header)
        self.assertIn("dispatch.workgroup_count_y = grid.y;", header)
        self.assertEqual(implementation.count("set_slice_dispatch_grid("), 2)
        self.assertNotIn("workgroup_count_y = wgc.y", implementation)

    def test_slice_correctness_does_not_require_profiling(self) -> None:
        source = _DYNAMIC_SHAPE_TEST.read_text()
        profile_available = source.split("bool slice_profile_available()", 1)[1].split(
            "void expect_slice_profile", 1
        )[0]
        boundary = source.split(
            "TEST(DynamicShape, SliceCrosses2dDispatchBoundary)", 1
        )[1].split("TEST(DynamicShape, SliceDualStoreWritesBothDestinations)", 1)[0]
        dual_store = source.split(
            "TEST(DynamicShape, SliceDualStoreWritesBothDestinations)", 1
        )[1].split("TEST(DynamicShape, ExpandCopyRejectsDynamicShapesAtLoad)", 1)[0]

        self.assertNotIn("timestamp queries unavailable", boundary)
        self.assertNotIn("timestamp queries unavailable", dual_store)
        for predicate in (
            'std::getenv("WEBGPU_TIMESTAMP_QUERY") != nullptr',
            "context != nullptr",
            "context->timestamp_supported",
            "context->querypool != nullptr",
        ):
            self.assertIn(predicate, profile_available)
        self.assertEqual(source.count("if (slice_profile_available())"), 2)

    def test_runs_codegen_pin_gate_before_fixture_exports(self) -> None:
        script = _ci_script()
        command = (
            "buck2 test " "fbcode//executorch/backends/webgpu/test:test_wgsl_codegen"
        )

        self.assertIn(
            "test_wgsl_codegen",
            _target_names((_BACKEND / "test/BUCK").read_text()),
        )
        self.assertEqual(script.count(command), 1)
        self.assertLess(script.index(command), script.index("# ── Exports"))

    def test_persistently_validates_wasm_names_without_claiming_products(
        self,
    ) -> None:
        script = _ci_script()
        invocation = (
            'bash "${SCRIPT_DIR}/test_gemma4_wasm_factory_contract.sh" '
            "--validate-names"
        )
        self.assertEqual(script.count(invocation), 1)
        self.assertNotIn("--verify-product", script)
        self.assertLess(script.index(invocation), script.index("# ── Exports"))

    def test_builds_and_runs_every_fixed_target(self) -> None:
        cmake = _CMAKE.read_text()
        script = _ci_script()
        required = {
            "webgpu_native_test",
            "webgpu_dispatch_order_test",
            "webgpu_scratch_buffer_test",
            "webgpu_update_cache_test",
            "webgpu_update_cache_state_test",
            "webgpu_index_test",
            "webgpu_dynamic_shape_test",
            "webgpu_dispatch_2d_test",
            "webgpu_compute_dispatch_test",
            "webgpu_execution_options_test",
            "webgpu_output_suppression_test",
            "webgpu_op_test_util_test",
            "webgpu_topk_test",
            "webgpu_scatter_test",
            "webgpu_q4gsw_m3_test",
        }

        self.assertEqual(set(_bash_array(script, "REQUIRED_TARGETS")), required)
        self.assertNotIn("not defined in this tree — skipping", script)
        self.assertIn('-DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"', script)
        self.assertIn("run_with_required_device env WEBGPU_TEST_SDPA_DIR", script)
        self.assertIn(
            "if ! grep -q '^WebGPU device acquired (native)$' " '<<<"${output}"; then',
            script,
        )
        for target in required:
            self.assertTrue(_cmake_defines(cmake, target), target)
            self.assertIn(f'"${{BIN_DIR}}/{target}"', script)

    def test_requires_symint_and_suppression_fixtures(self) -> None:
        script = _ci_script()

        self.assertIn(
            "export_output_suppression_models('${OUTPUT_SUPPRESSION_DIR}')", script
        )
        self.assertIn('WEBGPU_TEST_SYMINT_BLOB="${SYMINT_BLOB}"', script)
        for fixture in (
            "${SYMINT_BLOB}",
            "${OUTPUT_SUPPRESSION_DIR}/input.bin",
        ):
            self.assertIn(f'require_file "{fixture}"', script)

    def test_requires_dynamic_rope_fixture(self) -> None:
        script = _ci_script()

        self.assertIn("export_rope_hf_dynamic('${ROPE_HF_DIR}')", script)
        self.assertIn('WEBGPU_TEST_ROPE_HF_DIR="${ROPE_HF_DIR}"', script)
        self.assertIn('require_file "${ROPE_HF_DIR}/rope_hf_dynamic.pte"', script)

    def test_cat_2d_regressions_are_heavy_and_fail_closed(self) -> None:
        backend = pathlib.Path(__file__).parents[1]
        script = (backend / "scripts/test_webgpu_native_ci.sh").read_text()
        dynamic_test = (backend / "test/native/test_dynamic_shape.cpp").read_text()
        dispatch_test = (backend / "test/native/test_dispatch_2d.cpp").read_text()
        cases = (backend / "test/op_tests/cases.py").read_text()
        driver = (backend / "test/op_tests/op_test_driver.cpp").read_text()

        self.assertIn('require_file "${DYNAMIC_SHAPE_DIR}/dyn_cat_2d.pte"', script)
        self.assertIn(
            'WEBGPU_TEST_HEAVY=1 $PYTHON_EXECUTABLE -c "\n'
            "from executorch.backends.webgpu.test.ops.dynamic_shape."
            "test_dynamic_shape_export import export_dynamic_shape_cases",
            script,
        )
        self.assertIn("DynamicShape.CatCrosses2dDispatchBoundary", script)
        self.assertIn("[  PASSED  ] 3 tests.", script)
        self.assertIn('CAT_2D_TEST_DIR="/tmp/webgpu_cat_2d_test"', script)
        self.assertIn("WEBGPU_TEST_HEAVY=1 $PYTHON_EXECUTABLE", script)
        self.assertIn('--output "${CAT_2D_TEST_DIR}" --ops cat', script)
        self.assertIn(
            "run_with_required_device env WEBGPU_REQUIRE_DEVICE=1", script
        )

        self.assertIn("TEST(DynamicShape, CatCrosses2dDispatchBoundary)", dynamic_test)
        self.assertIn(
            "{kCat2dRows, kCat1dRows, kCat2dRows, kCat1dRows, kCat2dRows}",
            dynamic_test,
        )
        self.assertIn(
            "TEST(CatDispatchGrid, RestoresBothDimensionsAcrossResize)",
            dispatch_test,
        )
        self.assertIn('name="folded_2d_full_output"', cases)
        self.assertIn("inputs=((65536, 65), (65536, 1))", cases)
        self.assertIn("heavy=True", cases)
        self.assertIn('std::getenv("WEBGPU_REQUIRE_DEVICE")', driver)
        self.assertIn("required_device_failure_exit_code", driver)
        self.assertIn('std::printf("WebGPU device acquired (native)\\n")', driver)

    def test_exports_and_requires_topk_and_scatter_fixtures(self) -> None:
        script = _ci_script()

        self.assertIn(
            'EAGLE_TOPK_EAGER_RECEIPT="${TOPK_AUTHORITY}" $PYTHON_EXECUTABLE '
            "-m unittest",
            script,
        )
        self.assertIn(
            "executorch.backends.webgpu.test.ops.topk.test_topk.TestEagleTopKCpu"
            ".test_eager_reference_is_repeatable",
            script,
        )
        self.assertIn(
            "-m executorch.backends.webgpu.test.ops.topk.export_topk_artifacts",
            script,
        )
        self.assertIn('"${TOPK_DIR}" "${TOPK_AUTHORITY}"', script)
        self.assertIn(
            "-m executorch.backends.webgpu.test.ops.scatter.export_scatter_artifacts",
            script,
        )
        for fixture in (
            "${TOPK_AUTHORITY}",
            "${TOPK_DIR}/cases.txt",
            "${SCATTER_DIR}/cases.txt",
            "${SCATTER_DIR}/base.bin",
        ):
            self.assertIn(f'require_file "{fixture}"', script)

        self.assertIn('"${BIN_DIR}/webgpu_topk_test" "${TOPK_DIR}"', script)
        self.assertIn('"${BIN_DIR}/webgpu_scatter_test" "${SCATTER_DIR}"', script)

    def test_every_registered_source_exists(self) -> None:
        missing: list[str] = []
        for build_file in _BUILD_FILES:
            entries = _srcs_entries(build_file.read_text())
            self.assertNotEqual(
                entries, [], f"{build_file} registers no srcs; parser drifted"
            )
            for entry in entries:
                if not (build_file.parent / entry).is_file():
                    missing.append(f"{build_file}: {entry}")
        self.assertEqual(
            missing, [], f"registrations point at missing sources: {missing}"
        )

    def test_every_d10_test_source_is_registered(self) -> None:
        registered = _registered_sources()
        unregistered = [name for name in _D10_TEST_SOURCES if name not in registered]
        self.assertEqual(
            unregistered,
            [],
            f"no Buck srcs, CMake target, or CI invocation names: {unregistered}",
        )

    def test_committed_d10_test_sources_match_the_uncommitted_working_copy(
        self,
    ) -> None:
        """Strict where source control can still identify D10; else says so."""
        reported = _reported_test_sources()
        if reported is None:
            self.assertFalse(
                _sapling_is_usable(),
                "`sl status` failed inside a Sapling working copy, so the "
                "committed D10 test source list went unverified here",
            )
            return
        if _THIS_GATE not in reported:
            self.assertNotEqual(
                _sl_log_node(_THIS_GATE),
                "",
                "this gate is neither an uncommitted change nor a committed "
                "file, so the committed D10 test source list went unverified",
            )
            return
        self.assertEqual(
            reported,
            list(_D10_TEST_SOURCES),
            "the committed D10 test source list drifted from the working copy",
        )

    def test_mtp_command_contract_targets_are_defined(self) -> None:
        undefined = _undefined_labels(_MTP_BUCK_TEST_TARGETS)
        self.assertEqual(undefined, [], f"MTP labels are undefined: {undefined}")

    def test_plain_regression_command_contract_targets_are_defined(self) -> None:
        undefined = _undefined_labels(_PLAIN_REGRESSION_TARGETS)
        self.assertEqual(
            undefined, [], f"plain-regression labels are undefined: {undefined}"
        )

    def test_documented_cmake_build_targets_are_defined(self) -> None:
        cmake = _CMAKE.read_text()
        missing = [
            name for name in _CMAKE_BUILD_TARGETS if not _cmake_defines(cmake, name)
        ]
        self.assertEqual(
            missing, [], f"CMakeLists.txt defines no such target: {missing}"
        )

    def test_orphaned_test_targets_bzl_stays_a_pure_duplicate(self) -> None:
        buck = (_BACKEND / "test/BUCK").read_text()
        orphan = (_BACKEND / "test/targets.bzl").read_text()

        self.assertNotIn('load(":targets.bzl"', buck)
        self.assertEqual(
            _target_names(orphan) - _target_names(buck),
            set(),
            "test/BUCK does not load test/targets.bzl, so a target only the "
            "latter defines would never be built",
        )
