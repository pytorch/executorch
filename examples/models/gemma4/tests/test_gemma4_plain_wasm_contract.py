# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest

from pathlib import Path


class Gemma4PlainWasmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_root = Path(__file__).resolve().parents[1]
        cls.executorch_root = cls.model_root.parents[2]
        cls.runner_path = cls.model_root / "runner" / "gemma4_plain_wasm.cpp"
        cls.runner = (
            cls.runner_path.read_text(encoding="utf-8")
            if cls.runner_path.is_file()
            else ""
        )
        cls.backend_cmake = (
            cls.executorch_root / "backends" / "webgpu" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        cls.model_cmake = (cls.model_root / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )

    def test_production_runner_exists(self) -> None:
        self.assertTrue(self.runner_path.is_file())

    def test_compare_abi_is_exported(self) -> None:
        exports = set(
            re.findall(
                r"GEMMA4_WASM_EXPORT\s+(?:const char\*|void|int)\s+"
                r"(et_[a-z0-9_]+)\s*\(",
                self.runner,
            )
        )
        self.assertTrue(
            {
                "et_init",
                "et_load",
                "et_unload",
                "et_reset",
                "et_prefill_batch",
                "et_prefill_step",
                "et_step",
                "et_profile_enable",
                "et_profile",
                "et_get_last_prefill_token_count",
                "et_get_route_contract_version",
                "et_get_last_route_mask",
                "et_get_last_route_conflict_count",
            }.issubset(exports)
        )
        self.assertNotIn("et_set_variant", exports)

    def test_compact_token_output_is_long_not_float_logits(self) -> None:
        self.assertIn("output_tensor_meta(0)", self.runner)
        self.assertGreaterEqual(
            self.runner.count("ScalarType::Long"),
            4,
        )
        self.assertIn("const_data_ptr<int64_t>()", self.runner)
        self.assertNotIn("const_data_ptr<float>()", self.runner)
        self.assertIn("output.numel() != 1", self.runner)

    def test_load_requires_three_ordered_ptds(self) -> None:
        self.assertIn("kExpectedPtdCount = 3", self.runner)
        self.assertIn("ptd_paths.size() != kExpectedPtdCount", self.runner)
        self.assertIn("load_webgpu_model", self.runner)
        self.assertIn("std::move(ptd_paths)", self.runner)

    def test_reset_reloads_the_text_decoder_and_clears_observations(self) -> None:
        reset_match = re.search(
            r"GEMMA4_WASM_EXPORT\s+int\s+et_reset\(\)\s*\{(?P<body>.*?)\n\}",
            self.runner,
            re.DOTALL,
        )
        self.assertIsNotNone(reset_match)
        body = reset_match.group("body") if reset_match is not None else ""
        self.assertIn("unload_method(kMethodName)", body)
        self.assertIn("load_method(kMethodName)", body)
        self.assertIn("reset_runtime_observations()", body)

        observations_match = re.search(
            r"void\s+reset_runtime_observations\(\)\s*\{(?P<body>.*?)\n\}",
            self.runner,
            re.DOTALL,
        )
        self.assertIsNotNone(observations_match)
        observations = (
            observations_match.group("body")
            if observations_match is not None
            else ""
        )
        self.assertIn("querypool->reset(0)", observations)

    def test_production_runner_has_no_variant_ab_switch(self) -> None:
        self.assertNotIn("et_set_variant", self.runner)
        self.assertNotIn("WEBGPU_VARIANT_", self.runner)

    def test_runner_has_no_dashboard_or_local_artifact_dependency(self) -> None:
        forbidden = (
            "/home/",
            "localhost",
            "manifold",
            "webgpu-e2e",
            "webgpu_benchmark",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.runner.lower())

    def test_cmake_and_buck_own_the_runner(self) -> None:
        targets = (self.model_root / "targets.bzl").read_text(encoding="utf-8")
        for build_file in (self.backend_cmake, targets):
            with self.subTest(build_file=build_file):
                self.assertIn("runner/gemma4_plain_wasm.cpp", build_file)
                self.assertIn("webgpu_backend", build_file)
                self.assertIn("webgpu_model_loader", build_file)
        self.assertNotIn("gemma4_plain_wasm", self.model_cmake)

    def test_cmake_builds_a_compare_loadable_browser_module(self) -> None:
        start = self.backend_cmake.index("add_executable(\n    gemma4_plain_wasm")
        end_marker = '"${CMAKE_CURRENT_BINARY_DIR}/browser_gemma4_plain"'
        end = self.backend_cmake.index(end_marker, start) + len(end_marker)
        cmake = self.backend_cmake[start:end]
        required_link_contract = (
            "--use-port=emdawnwebgpu",
            "-sASYNCIFY",
            "-sALLOW_MEMORY_GROWTH=1",
            "-sMAXIMUM_MEMORY=4GB",
            "-sFORCE_FILESYSTEM=1",
            "--no-entry",
            "-sSTACK_SIZE=8388608",
            "-sASYNCIFY_STACK_SIZE=1048576",
            "-sMODULARIZE=1",
            "-sEXPORT_NAME=createWebGPULlama",
            'OUTPUT_NAME "webgpu_llama"',
            "browser_gemma4_plain",
        )
        for option in required_link_contract:
            with self.subTest(option=option):
                self.assertIn(option, cmake)
        self.assertNotIn("-sNO_ENTRY", cmake)

        runtime_methods = re.search(
            r'-sEXPORTED_RUNTIME_METHODS=([^"\s]+)', cmake
        )
        self.assertIsNotNone(runtime_methods)
        self.assertEqual(
            set(runtime_methods.group(1).split(",")) if runtime_methods else set(),
            {"ccall", "cwrap", "FS", "HEAP32"},
        )
        expected_functions = {
            "_et_init",
            "_et_load",
            "_et_unload",
            "_et_reset",
            "_et_step",
            "_et_prefill_step",
            "_et_prefill_batch",
            "_et_profile_enable",
            "_et_profile",
            "_et_get_last_prefill_token_count",
            "_et_get_route_contract_version",
            "_et_get_last_route_mask",
            "_et_get_last_route_conflict_count",
            "_malloc",
            "_free",
        }
        exported_functions = re.search(r'-sEXPORTED_FUNCTIONS=([^"\s]+)', cmake)
        self.assertIsNotNone(exported_functions)
        self.assertEqual(
            set(exported_functions.group(1).split(","))
            if exported_functions
            else set(),
            expected_functions,
        )
