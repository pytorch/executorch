# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import re
import shlex
import unittest


def _bash_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(name)}=\((.*?)\)", source, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise AssertionError(f"{name} Bash array not found")
    return shlex.split(match.group(1))


class TestNativeCIContract(unittest.TestCase):
    def test_builds_and_runs_every_fixed_target(self) -> None:
        backend = pathlib.Path(__file__).parents[1]
        cmake = (backend / "CMakeLists.txt").read_text()
        script = (backend / "scripts/test_webgpu_native_ci.sh").read_text()
        required = {
            "webgpu_native_test",
            "webgpu_dispatch_order_test",
            "webgpu_scratch_buffer_test",
            "webgpu_update_cache_test",
            "webgpu_index_test",
            "webgpu_dynamic_shape_test",
            "webgpu_dispatch_2d_test",
            "webgpu_compute_dispatch_test",
            "webgpu_execution_options_test",
            "webgpu_output_suppression_test",
            "webgpu_op_test_util_test",
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
            self.assertIn(target, cmake)
            self.assertIn(f'"${{BIN_DIR}}/{target}"', script)

    def test_requires_symint_and_suppression_fixtures(self) -> None:
        script = (
            pathlib.Path(__file__).parents[1] / "scripts/test_webgpu_native_ci.sh"
        ).read_text()

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
        script = (
            pathlib.Path(__file__).parents[1] / "scripts/test_webgpu_native_ci.sh"
        ).read_text()

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
