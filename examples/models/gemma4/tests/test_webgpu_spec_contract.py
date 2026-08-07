# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tempfile
import unittest

from pathlib import Path
from typing import Mapping, Sequence


GEMMA4_ANCHOR = "examples/models/gemma4/targets.bzl"
SOURCE_ROOT_ENV = "EXECUTORCH_SOURCE_ROOT"
SPEC_RUNNER_HEADER = "examples/models/gemma4/runner/gemma4_spec_runner.h"
SPEC_RUNNER_SOURCE = "examples/models/gemma4/runner/gemma4_spec_runner.cpp"
SPEC_WASM_SOURCE = "examples/models/gemma4/runner/gemma4_spec_wasm.cpp"
GEMMA4_RUNNER_HEADER = "examples/models/gemma4/runner/gemma4_runner.h"
GEMMA4_RUNNER_SOURCE = "examples/models/gemma4/runner/gemma4_runner.cpp"
GEMMA4_TARGETS = "examples/models/gemma4/targets.bzl"
GEMMA4_CMAKE = "examples/models/gemma4/CMakeLists.txt"
GEMMA4_README = "examples/models/gemma4/README.md"
WEBGPU_CMAKE = "backends/webgpu/CMakeLists.txt"
WASM_FACTORY_CONTRACT = "backends/webgpu/scripts/test_gemma4_wasm_factory_contract.sh"
WEBGPU_BACKEND_SOURCE = "backends/webgpu/runtime/WebGPUBackend.cpp"
WEBGPU_EXECUTION_OPTIONS_SOURCE = "backends/webgpu/runtime/WebGPUExecutionOptions.cpp"

EXPECTED_WASM_EXPORTS = (
    "et_init",
    "et_load",
    "et_unload",
    "et_reset",
    "et_prefill_batch",
    "et_prefill_step",
    "et_step",
    "et_mtp_execute_count",
    "et_mtp_accepted_drafts",
    "et_mtp_buffered_tokens",
    "et_mtp_execute",
    "et_mtp_execution_attestation",
    "et_profile_enable",
    "et_profile",
)
EXPECTED_EXPORTED_FUNCTIONS: tuple[str, ...] = tuple(
    f"_{symbol}" for symbol in EXPECTED_WASM_EXPORTS
) + ("_malloc", "_free")
EXPECTED_RUNTIME_METHODS = ("ccall", "cwrap", "FS", "HEAP32")

REQUIRED_RUNNER_SYMBOLS = (
    "Gemma4SpecRunner::accepted_drafts",
    "Gemma4SpecRunner::buffered_tokens",
    "Gemma4SpecRunner::execute",
    "Gemma4SpecRunner::execute_count",
    "Gemma4SpecRunner::generate",
    "Gemma4SpecRunner::is_loaded",
    "Gemma4SpecRunner::load",
    "Gemma4SpecRunner::prefill",
    "Gemma4SpecRunner::prefill_step",
    "Gemma4SpecRunner::profile_json",
    "Gemma4SpecRunner::reset",
    "Gemma4SpecRunner::set_profiling_enabled",
    "Gemma4SpecRunner::step",
    "Gemma4SpecRunner::unload",
)

XNNPACK_SYMBOLS = (
    "weight_cache_option_key",
    "workspace_sharing_mode_option_key",
    "xnnpack_backend_key",
)

PUBLIC_GEMMA4_RUNNER_API = (
    "Gemma4Runner",
    "load",
    "is_loaded",
    "generate",
    "generate",
    "generate_text",
    "generate_text",
    "generate_vision",
    "generate_vision",
    "reset",
)

RESET_LANDMARKS = (
    "impl_->arm_profile();",
    "impl_->clear_controller_state();",
    "impl_->method_fresh = false;",
    "impl_->method_fresh = impl_->method_healthy;",
    "impl_->method_healthy = false;",
    "->load_method(",
    "->unload_method(",
)
EXPECTED_RESET_ORDER = (
    "impl_->method_healthy = false;",
    "impl_->method_fresh = false;",
    "impl_->clear_controller_state();",
    "impl_->arm_profile();",
    "impl_->clear_controller_state();",
    "->unload_method(",
    "impl_->method_healthy = false;",
    "impl_->method_fresh = false;",
    "->load_method(",
    "impl_->method_fresh = impl_->method_healthy;",
)
EXPECTED_HEALTH_LATCH: Mapping[str, tuple[str, ...]] = {
    "execute": ("false",) * 10,
    "generate": ("false",),
    "load": ("false", "false", "false", "true"),
    "reset": ("false", "false"),
    "step": ("false",),
    "unload": ("false",),
}

_WASM_EXPORT_PATTERN: re.Pattern[str] = re.compile(
    r"^ET_WASM_EXPORT\s+[A-Za-z_][\w:*&<>\s]*?\b(et_[a-z0-9_]+)\s*\(", re.M | re.S
)
_DEFINITION_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z_][^\n;{}]*\bGemma4SpecRunner::(\w+)\(", re.M
)
_HEALTH_PATTERN: re.Pattern[str] = re.compile(r"impl_->method_healthy = ([^;]+);")
_BLOCK_COMMENT_PATTERN: re.Pattern[str] = re.compile(r"/\*.*?\*/", re.S)


def _root_candidates() -> list[tuple[str, Path | None]]:
    override = os.environ.get(SOURCE_ROOT_ENV)
    try:
        package = importlib.util.find_spec("executorch")
    except (ImportError, ValueError):
        package = None
    staged = list(package.submodule_search_locations or ()) if package else []
    here = Path(__file__).resolve()
    walked = next(
        (parent for parent in here.parents if (parent / GEMMA4_ANCHOR).is_file()), None
    )
    return [
        (f"${SOURCE_ROOT_ENV}", Path(override) if override else None),
        ("`executorch` package runfile", Path(staged[0]) if staged else None),
        (f"__file__ walk above {here}", walked),
    ]


def _source_root() -> Path:
    attempted: list[str] = []
    for strategy, candidate in _root_candidates():
        attempted.append(f"{strategy} -> {candidate}")
        if candidate is not None and (candidate / GEMMA4_ANCHOR).is_file():
            return candidate
    raise FileNotFoundError(
        f"no ExecuTorch source root containing {GEMMA4_ANCHOR}; "
        f"tried {'; '.join(attempted)}"
    )


def _read(relative: str) -> str:
    path = _source_root() / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing source under test: {path}")
    return path.read_text(encoding="utf-8")


def _verify_product(
    source: str, expected_factory: str, expected_output_stem: str
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        javascript = Path(directory) / "product.js"
        javascript.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                "bash",
                str(_source_root() / WASM_FACTORY_CONTRACT),
                "--verify-product",
                str(javascript),
                expected_factory,
                expected_output_stem,
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def wasm_exports(source: str) -> list[str]:
    return _WASM_EXPORT_PATTERN.findall(source)


def cmake_exported_functions(cmake: str) -> list[str]:
    match = re.search(r"-sEXPORTED_FUNCTIONS=\[([^\]]*)\]", cmake)
    if match is None:
        raise AssertionError("gemma4_spec_browser declares no -sEXPORTED_FUNCTIONS")
    return re.findall(r"'([^']+)'", match.group(1))


def cmake_runtime_methods(cmake: str) -> list[str]:
    match = re.search(r"-sEXPORTED_RUNTIME_METHODS=\[([^\]]*)\]", cmake)
    if match is None:
        raise AssertionError(
            "gemma4_spec_browser declares no -sEXPORTED_RUNTIME_METHODS"
        )
    return re.findall(r"'([^']+)'", match.group(1))


def required_symbol_census(sources: Mapping[str, str]) -> set[str]:
    census: set[str] = set()
    for text in sources.values():
        census.update(wasm_exports(text))
        census.update(
            f"Gemma4SpecRunner::{name}" for name in _DEFINITION_PATTERN.findall(text)
        )
    return census


def missing_required_symbols(sources: Mapping[str, str]) -> set[str]:
    required = set(EXPECTED_WASM_EXPORTS) | set(REQUIRED_RUNNER_SYMBOLS)
    return required - required_symbol_census(sources)


def _brace_body(text: str, start: int) -> str:
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1]
    raise AssertionError(f"unbalanced braces after offset {start}")


def definition_bodies(source: str) -> dict[str, str]:
    return {
        match.group(1): _brace_body(source, match.end())
        for match in _DEFINITION_PATTERN.finditer(source)
    }


def wasm_definition_bodies(source: str) -> dict[str, str]:
    return {
        match.group(1): _brace_body(source, match.end())
        for match in _WASM_EXPORT_PATTERN.finditer(source)
    }


def ordered_landmarks(body: str, landmarks: Sequence[str]) -> list[str]:
    hits: list[tuple[int, str]] = []
    for landmark in landmarks:
        start = 0
        while True:
            index = body.find(landmark, start)
            if index < 0:
                break
            hits.append((index, landmark))
            start = index + 1
    return [landmark for _, landmark in sorted(hits)]


def bzl_rule(text: str, name: str) -> str:
    start = text.index(f'name = "{name}",')
    return text[start : text.index("\n    )\n", start)]


def public_api(header: str) -> list[str]:
    stripped = _BLOCK_COMMENT_PATTERN.sub("", header)
    section = stripped[stripped.index(" public:") : stripped.index(" private:")]
    return [
        match.group(1)
        for match in re.finditer(
            r"^\s{2}(?:[A-Za-z_][\w:<>,\s*&]*?\s)?([A-Za-z_]\w*)\s*\(",
            section,
            re.M,
        )
    ]


class BrowserAbiContractTest(unittest.TestCase):
    def test_adapter_defines_exactly_the_reviewed_export_list(self) -> None:
        self.assertEqual(
            wasm_exports(_read(SPEC_WASM_SOURCE)), list(EXPECTED_WASM_EXPORTS)
        )

    def test_cmake_exports_the_adapter_symbols_plus_the_allocator(self) -> None:
        exported = cmake_exported_functions(_read(WEBGPU_CMAKE))
        self.assertEqual(exported, list(EXPECTED_EXPORTED_FUNCTIONS))
        self.assertEqual(len(exported), 16)
        self.assertEqual(
            [symbol[1:] for symbol in exported if symbol.startswith("_et_")],
            list(EXPECTED_WASM_EXPORTS),
        )

    def test_cmake_exports_the_reviewed_browser_runtime_methods(self) -> None:
        self.assertEqual(
            cmake_runtime_methods(_read(WEBGPU_CMAKE)),
            list(EXPECTED_RUNTIME_METHODS),
        )

    def test_link_mutant_omitting_the_adapter_fails_the_census(self) -> None:
        complete = {
            SPEC_WASM_SOURCE: _read(SPEC_WASM_SOURCE),
            SPEC_RUNNER_SOURCE: _read(SPEC_RUNNER_SOURCE),
        }
        self.assertEqual(missing_required_symbols(complete), set())
        without_adapter = {SPEC_RUNNER_SOURCE: complete[SPEC_RUNNER_SOURCE]}
        self.assertEqual(
            missing_required_symbols(without_adapter), set(EXPECTED_WASM_EXPORTS)
        )
        without_runner = {SPEC_WASM_SOURCE: complete[SPEC_WASM_SOURCE]}
        self.assertEqual(
            missing_required_symbols(without_runner), set(REQUIRED_RUNNER_SYMBOLS)
        )
        self.assertEqual(
            missing_required_symbols({}),
            set(EXPECTED_WASM_EXPORTS) | set(REQUIRED_RUNNER_SYMBOLS),
        )

    def test_adapter_pins_the_method_name_and_tensor_data_path_count(self) -> None:
        source = _read(SPEC_WASM_SOURCE)
        self.assertIn('constexpr const char* kMethodName = "k2_round";', source)
        self.assertIn("constexpr size_t kExpectedTensorDataPaths = 3;", source)
        self.assertIn("if (paths.size() != kExpectedTensorDataPaths)", source)

    def test_execution_attestation_reads_the_backend_on_every_call(self) -> None:
        bodies = wasm_definition_bodies(_read(SPEC_WASM_SOURCE))
        self.assertIn("et_mtp_execution_attestation", bodies)
        body = bodies.get("et_mtp_execution_attestation", "")
        self.assertEqual(body.count("webgpu_backend_execution_attestation_json()"), 1)
        self.assertIn(
            "execution_attestation_json = webgpu_backend_execution_attestation_json();",
            body,
        )
        self.assertIn("return execution_attestation_json.c_str();", body)

    def test_attestation_reports_observed_pass_and_submit_counts(self) -> None:
        backend = _read(WEBGPU_BACKEND_SOURCE)
        serializer = _read(WEBGPU_EXECUTION_OPTIONS_SOURCE)
        self.assertIn("last_execution_graph->execution_attestation_json()", backend)
        self.assertIn('\\"encodedComputePasses\\":', serializer)
        self.assertIn('\\"queueSubmitCount\\":', serializer)


class SpecRunnerSourceContractTest(unittest.TestCase):
    def test_controller_progression_and_bonus_seeding_are_pinned(self) -> None:
        header = _read(SPEC_RUNNER_HEADER)
        self.assertIn(
            "decision.next_position = start_position + output.match_count + 1;",
            header,
        )
        self.assertIn("decision.next_seed = output.bonus;", header)
        self.assertIn("decision.selected.push_back(output.bonus);", header)

    def test_self_consistency_guard_is_pinned(self) -> None:
        header = _read(SPEC_RUNNER_HEADER)
        self.assertIn(
            "if (output.match_count != expected_matches || "
            "!valid_token(output.bonus) ||",
            header,
        )
        self.assertIn(
            "output.bonus != output.target_greedy[output.match_count]) {", header
        )
        self.assertIn(
            "if (start_position < 2 || token_budget == 0 || vocab_size <= 0 ||",
            header,
        )
        self.assertIn("output.match_count < 0 || output.match_count > 2 ||", header)
        self.assertIn("!std::isfinite(output.state_probe)) {", header)

    def test_config_defaults_match_the_export_contract(self) -> None:
        header = _read(SPEC_RUNNER_HEADER)
        for default in (
            "int64_t vocab_size = 262144;",
            "int64_t max_input_length = 512;",
            "int64_t target_capacity = 8960;",
            "int64_t donor_capacity = 8960;",
            'std::string method_name = "k2_round";',
            "int64_t vocab_size = 262144) {",
        ):
            with self.subTest(default=default):
                self.assertIn(default, header)

    def test_load_pins_the_four_input_five_output_vulkan_abi(self) -> None:
        source = _read(SPEC_RUNNER_SOURCE)
        self.assertIn("meta.num_inputs() != 4 || meta.num_outputs() != 5 ||", source)
        self.assertIn(
            "meta.num_backends() != 1 || meta.num_instructions() != 1 ||",
            source,
        )
        self.assertIn('std::string_view(backend.get()) != "VulkanBackend"', source)
        self.assertIn("methods->size() != 1 ||", source)

    def test_round_execute_requires_three_rows_and_a_start_aligned_donor(
        self,
    ) -> None:
        source = _read(SPEC_RUNNER_SOURCE)
        self.assertIn("(is_round && input_ids.size() != 3)", source)
        self.assertIn(
            "if ((is_round && (donor_length != start_position || donor_length < 2))",
            source,
        )
        self.assertIn("if (execution->size() != 5) {", source)

    def test_execute_cannot_request_uncertified_single_compute_pass(self) -> None:
        source = _read(SPEC_RUNNER_SOURCE)
        body = definition_bodies(source)["execute"]
        for forbidden in (
            "WebGPUExecutionOptions",
            "single_compute_pass",
            "with_webgpu_execution_options(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn(
            "auto execution = impl_->module->execute(\n"
            "      impl_->config.method_name,",
            body,
        )

    def test_reset_makes_the_next_attestation_fresh(self) -> None:
        reset = definition_bodies(_read(SPEC_RUNNER_SOURCE))["reset"]
        unload = reset.index("->unload_method(")
        reload = reset.index("->load_method(")
        self.assertLess(unload, reload)

        backend = _read(WEBGPU_BACKEND_SOURCE)
        destroy_start = backend.index("void WebGPUBackend::destroy(")
        destroy = _brace_body(backend, destroy_start)
        self.assertIn("if (last_execution_graph == graph)", destroy)
        self.assertIn("last_execution_graph = nullptr;", destroy)

        bodies = wasm_definition_bodies(_read(SPEC_WASM_SOURCE))
        self.assertIn("et_mtp_execution_attestation", bodies)
        attestation = bodies.get("et_mtp_execution_attestation", "")
        self.assertIn("webgpu_backend_execution_attestation_json()", attestation)

    def test_reset_clears_state_then_unloads_then_reloads(self) -> None:
        body = definition_bodies(_read(SPEC_RUNNER_SOURCE))["reset"]
        self.assertEqual(
            tuple(ordered_landmarks(body, RESET_LANDMARKS)), EXPECTED_RESET_ORDER
        )
        self.assertIn(
            "error == Error::Ok && verify_context(impl_->context.get())", body
        )

    def test_method_healthy_latch_is_cleared_only_by_reset_and_load(self) -> None:
        source = _read(SPEC_RUNNER_SOURCE)
        bodies = definition_bodies(source)
        observed = {
            name: tuple(_HEALTH_PATTERN.findall(body))
            for name, body in bodies.items()
            if _HEALTH_PATTERN.search(body)
        }
        self.assertEqual(observed, EXPECTED_HEALTH_LATCH)
        self.assertEqual(len(_HEALTH_PATTERN.findall(source)), 19)

    def test_unload_surfaces_a_failed_method_unload(self) -> None:
        body = definition_bodies(_read(SPEC_RUNNER_SOURCE))["unload"]
        self.assertIn("return method_unloaded ? Error::Ok : Error::Internal;", body)
        self.assertIn("impl_->clear_controller_state();", body)
        self.assertIn("destroy_webgpu_context(*impl_->context);", body)

    def test_is_loaded_requires_a_module_and_a_healthy_method(self) -> None:
        body = definition_bodies(_read(SPEC_RUNNER_SOURCE))["is_loaded"]
        self.assertIn(
            "return impl_->module != nullptr && impl_->method_healthy &&", body
        )
        self.assertIn(
            "impl_->context != nullptr && verify_context(impl_->context.get())", body
        )


class XnnpackPreservationTest(unittest.TestCase):
    def test_spec_sources_reference_no_xnnpack_symbol(self) -> None:
        control = _read(GEMMA4_RUNNER_SOURCE)
        for symbol in XNNPACK_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, control)
        for relative in (SPEC_RUNNER_HEADER, SPEC_RUNNER_SOURCE, SPEC_WASM_SOURCE):
            text = _read(relative)
            for symbol in XNNPACK_SYMBOLS + ("xnnpack", "XNNPACK"):
                with self.subTest(source=relative, symbol=symbol):
                    self.assertNotIn(symbol, text)

    def test_spec_targets_declare_no_xnnpack_dependency(self) -> None:
        targets = _read(GEMMA4_TARGETS)
        for name in ("gemma4_spec_runner", "gemma4_spec_wasm_adapter"):
            with self.subTest(target=name):
                self.assertNotIn("xnnpack", bzl_rule(targets, name))

    def test_public_gemma4_runner_target_is_unchanged(self) -> None:
        rule = bzl_rule(_read(GEMMA4_TARGETS), "gemma4_runner")
        self.assertIn('srcs = [\n            "runner/gemma4_runner.cpp",\n', rule)
        for header in (
            "runner/gemma4_runner.h",
            "runner/gemma4_stats.h",
            "runner/generation_config.h",
        ):
            with self.subTest(header=header):
                self.assertIn(f'"{header}",', rule)
        self.assertIn('"//executorch/backends/xnnpack:xnnpack_interface",', rule)
        self.assertIn("deps = _KERNEL_BACKEND_DEPS + [", rule)

    def test_public_gemma4_runner_api_is_unchanged(self) -> None:
        self.assertEqual(
            public_api(_read(GEMMA4_RUNNER_HEADER)), list(PUBLIC_GEMMA4_RUNNER_API)
        )


class SpecBuildContractTest(unittest.TestCase):
    def test_spec_browser_compiles_the_adapter_runner_and_links_the_loader(
        self,
    ) -> None:
        cmake = _read(WEBGPU_CMAKE)
        start = cmake.index("add_executable(\n    gemma4_spec_browser")
        block = cmake[start : cmake.index("-sEXPORTED_FUNCTIONS", start)]
        for fragment in (
            "examples/models/gemma4/runner/gemma4_spec_runner.cpp",
            "examples/models/gemma4/runner/gemma4_spec_wasm.cpp",
            "webgpu_backend webgpu_model_loader",
            "extension_tensor",
            "--use-port=emdawnwebgpu",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, block)

    def test_mtp_target_consumes_validated_factory_and_output_cache_strings(
        self,
    ) -> None:
        cmake = _read(WEBGPU_CMAKE)
        plain_start = cmake.index("add_executable(\n    gemma4_plain_wasm")
        spec_start = cmake.index("add_executable(\n    gemma4_spec_browser")
        plain = cmake[plain_start:spec_start]
        spec = cmake[spec_start : cmake.index("\nendif()", spec_start)]
        self.assertIn(
            "set(GEMMA4_SPEC_WASM_EXPORT_NAME\n"
            '      "createGemma4Mtp"\n'
            "      CACHE STRING",
            cmake,
        )
        self.assertIn(
            "set(GEMMA4_SPEC_WASM_OUTPUT_NAME\n"
            '      "gemma4_mtp"\n'
            "      CACHE STRING",
            cmake,
        )
        self.assertIn("include(cmake/ValidateGemma4WasmNames.cmake)", cmake)
        self.assertIn(
            "validate_gemma4_wasm_names(\n"
            "    GEMMA4_SPEC_WASM_EXPORT_NAME GEMMA4_SPEC_WASM_OUTPUT_NAME\n"
            "  )",
            cmake,
        )
        self.assertIn("-sEXPORT_NAME=${GEMMA4_SPEC_WASM_EXPORT_NAME}", spec)
        self.assertIn('OUTPUT_NAME "${GEMMA4_SPEC_WASM_OUTPUT_NAME}"', spec)
        self.assertNotIn("GEMMA4_SPEC_WASM_EXPORT_NAME", plain)
        self.assertNotIn("GEMMA4_SPEC_WASM_OUTPUT_NAME", plain)
        self.assertIn("-sEXPORT_NAME=createWebGPULlama", plain)
        self.assertIn('OUTPUT_NAME "webgpu_llama"', plain)

    def test_readme_builds_distinct_wall_and_profile_products(self) -> None:
        readme = _read(GEMMA4_README)
        wall_start = readme.index('emcmake cmake "${COMMON[@]}" -B "$WALL_BUILD"')
        profile_start = readme.index('emcmake cmake "${COMMON[@]}" -B "$PROFILE_BUILD"')
        wall = readme[wall_start:profile_start]
        profile = readme[profile_start : readme.index("\n```", profile_start)]
        for block, required, forbidden in (
            (
                wall,
                (
                    "-DEXECUTORCH_BUILD_WEBGPU_PROFILING=OFF",
                    "-DGEMMA4_SPEC_WASM_EXPORT_NAME=createGemma4Mtp",
                    "-DGEMMA4_SPEC_WASM_OUTPUT_NAME=gemma4_mtp",
                ),
                ("PROFILING=ON", "createGemma4MtpProfile", "gemma4_mtp_profile"),
            ),
            (
                profile,
                (
                    "-DEXECUTORCH_BUILD_WEBGPU_PROFILING=ON",
                    "-DGEMMA4_SPEC_WASM_EXPORT_NAME=createGemma4MtpProfile",
                    "-DGEMMA4_SPEC_WASM_OUTPUT_NAME=gemma4_mtp_profile",
                ),
                ("PROFILING=OFF", "=createGemma4Mtp\n", "=gemma4_mtp\n"),
            ),
        ):
            for fragment in required:
                with self.subTest(fragment=fragment):
                    self.assertIn(fragment, block)
            for fragment in forbidden:
                with self.subTest(forbidden=fragment):
                    self.assertNotIn(fragment, block)
            self.assertIn("--target gemma4_plain_wasm gemma4_spec_browser", block)

        for fragment in (
            "plain-profile-recipe.json",
            "browser_gemma4_mtp/gemma4_mtp_profile.js",
            "browser_gemma4_mtp/gemma4_mtp_profile.wasm",
            "--plain-profile-javascript",
            "--plain-profile-wasm",
            "--plain-profile-recipe",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)

    def test_product_verifier_binds_factory_and_requested_wasm(self) -> None:
        accepted = _verify_product(
            "var createGemma4Mtp = async function(options) {"
            "options.locateFile('gemma4_mtp.wasm');};",
            "createGemma4Mtp",
            "gemma4_mtp",
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

        mutations = {
            "wrong factory": (
                "var createGemma4MtpProfile = async function(options) {"
                "options.locateFile('gemma4_mtp.wasm');};",
                "createGemma4Mtp",
                "gemma4_mtp",
            ),
            "wrong wasm": (
                "var createGemma4Mtp = async function(options) {"
                "options.locateFile('wrong.wasm');};",
                "createGemma4Mtp",
                "gemma4_mtp",
            ),
            "extra factory": (
                "var createGemma4Mtp = async function(options) {"
                "options.locateFile('gemma4_mtp.wasm');};"
                "var createWebGPULlama = function() {};",
                "createGemma4Mtp",
                "gemma4_mtp",
            ),
        }
        for label, (source, factory, output_stem) in mutations.items():
            with self.subTest(label=label):
                rejected = _verify_product(source, factory, output_stem)
                self.assertNotEqual(rejected.returncode, 0)

    def test_spec_browser_has_the_full_production_link_closure(self) -> None:
        cmake = _read(WEBGPU_CMAKE)
        start = cmake.index("add_executable(\n    gemma4_spec_browser")
        block = cmake[start : cmake.index("\nendif()", start)]
        for fragment in (
            "-fexceptions",
            '"--use-port=emdawnwebgpu"',
            '"-sASYNCIFY"',
            '"-sALLOW_MEMORY_GROWTH=1"',
            '"-sMAXIMUM_MEMORY=4GB"',
            '"-sFORCE_FILESYSTEM=1"',
            '"--no-entry"',
            "'HEAP32'",
            '"-sSTACK_SIZE=8388608"',
            '"-sASYNCIFY_STACK_SIZE=1048576"',
            '"-sMODULARIZE=1"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, block)
        self.assertNotIn("-sNO_ENTRY", block)

    def test_native_spec_runner_is_guarded_by_the_loader_target(self) -> None:
        cmake = _read(GEMMA4_CMAKE)
        self.assertIn("if(TARGET webgpu_backend AND TARGET webgpu_model_loader)", cmake)
        self.assertIn(
            "add_library(gemma4_spec_runner runner/gemma4_spec_runner.cpp)", cmake
        )
