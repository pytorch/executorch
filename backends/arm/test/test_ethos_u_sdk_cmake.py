# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess  # nosec B404 - runs trusted local CMake, Git, and Bash tools
import sys
import tempfile
import unittest
from pathlib import Path


ARM_ROOT = Path(__file__).resolve().parents[1]
CMAKE = shutil.which("cmake")
SDK_FILES = (
    "core_platform/targets/corstone-300/CMakeLists.txt",
    "core_platform/targets/corstone-320/CMakeLists.txt",
    "core_software/CMakeLists.txt",
    "core_software/core_driver/include/ethosu_driver.h",
    "core_software/cmsis_6/CMSIS/Core/Include/cmsis_compiler.h",
    "core_software/Cortex_DFP/ARM.Cortex_DFP.pdsc",
    "core_software/cmsis-nn/Include/arm_nnfunctions.h",
    "core_software/cmsis-view/EventRecorder/Source/EventRecorder.c",
)


@unittest.skipUnless(CMAKE, "CMake is required")
class TestEthosUSDKCMake(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ethos sdk ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / "sdk with spaces"
        for name in SDK_FILES:
            path = self.sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.event_recorder = self.sdk / SDK_FILES[-1]
        (self.sdk / "core_software/Cortex_DFP/Device").mkdir()

    def configure(self, commands, fetch_script="raise SystemExit(23)\n"):
        (self.sdk / "fetch_externals.py").write_text(
            "from pathlib import Path\n" "Path('fetch-called').touch()\n" + fetch_script
        )
        (self.root / "CMakeLists.txt").write_text(
            f"""
cmake_minimum_required(VERSION 3.19)
project(ethos_sdk_test NONE)
include("{ARM_ROOT}/scripts/corstone_utils.cmake")
# Keep the real external-fetch process, but avoid network and Git mutations.
function(patch_ethos_u_repo REPO_PATH)
  file(APPEND "{self.root}/patched.txt" "${{REPO_PATH}}\n")
endfunction()
set(sdk "{self.sdk}")
{commands}
"""
        )
        return subprocess.run(  # nosec B603 - fixed local test project and arguments
            [
                CMAKE,
                "-S",
                str(self.root),
                "-B",
                str(self.root / "build"),
                f"-DETHOS_SDK_PATH:PATH={self.sdk}",
                f"-DFETCHCONTENT_SOURCE_DIR_ETHOS_U:PATH={self.sdk}",
                f"-DPython3_EXECUTABLE={sys.executable}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_complete_sdk_does_not_fetch(self):
        result = self.configure(
            """
arm_ethos_u_default_fetch("${sdk}" fetch)
if(fetch)
  message(FATAL_ERROR "Complete SDK should default to no fetch")
endif()
arm_ensure_ethos_u_content("${sdk}" unused ON)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())
        self.assertFalse((self.root / "patched.txt").exists())

    def test_each_missing_dependency_enables_fetch(self):
        for name in SDK_FILES:
            with self.subTest(name=name):
                path = self.sdk / name
                path.unlink()
                result = self.configure(
                    """
arm_ethos_u_default_fetch("${sdk}" fetch)
if(NOT fetch)
  message(FATAL_ERROR "Incomplete SDK should default to fetch")
endif()
"""
                )
                path.touch()
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_driver_only_sdk_does_not_fetch(self):
        for name in SDK_FILES:
            if name != "core_software/core_driver/include/ethosu_driver.h":
                (self.sdk / name).unlink()
        result = self.configure(
            """
arm_ethos_u_default_fetch("${sdk}" fetch DRIVER_ONLY)
if(fetch)
  message(FATAL_ERROR "Core driver is sufficient for the backend library")
endif()
arm_ensure_ethos_u_content("${sdk}" unused OFF DRIVER_ONLY)
arm_ensure_ethos_u_content("${sdk}" unused ON DRIVER_ONLY)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())
        self.assertFalse((self.root / "patched.txt").exists())

    def test_single_platform_sdk_does_not_fetch(self):
        (self.sdk / "core_software/Cortex_DFP/ARM.Cortex_DFP.pdsc").unlink()
        (self.root / "driver.c").touch()
        for system_config, memory_mode, unused_platform in (
            ("Ethos_U55_High_End_Embedded", "Shared_Sram", "corstone-320"),
            ("Ethos_U65_High_End", "Dedicated_Sram_384KB", "corstone-320"),
            ("Ethos_U85_SYS_DRAM_Mid", "Dedicated_Sram_384KB", "corstone-300"),
        ):
            with self.subTest(system_config=system_config):
                path = self.sdk / "core_platform/targets" / unused_platform
                shutil.rmtree(path)
                result = self.configure(
                    f"""
arm_ethos_u_default_fetch("${{sdk}}" fetch SYSTEM_CONFIG {system_config})
if(fetch)
  message(FATAL_ERROR "Unused platform and dependencies should not enable fetch")
endif()
arm_ensure_ethos_u_content("${{sdk}}" unused OFF SYSTEM_CONFIG {system_config})
arm_ensure_ethos_u_content("${{sdk}}" unused ON SYSTEM_CONFIG {system_config})
include("{ARM_ROOT}/cmake/ArmRunnerUtils.cmake")
enable_language(C)
set(CMAKE_SKIP_INSTALL_RULES ON)
add_library(ethosu_core_driver STATIC "{self.root}/driver.c")
add_library(ethosu_target_common INTERFACE)
arm_runner_configure_ethos_u_platform(
  SDK_PATH "${{sdk}}" SYSTEM_CONFIG {system_config} MEMORY_MODE {memory_mode}
)
if(FETCH_ETHOS_U_CONTENT)
  message(FATAL_ERROR "Runner should leave fetching disabled for a usable SDK")
endif()
"""
                )
                path.mkdir()
                (path / "CMakeLists.txt").touch()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((self.sdk / "fetch-called").exists())
                self.assertFalse((self.root / "patched.txt").exists())

    def test_runner_accepts_external_dependencies(self):
        for directory in ("cmsis_6", "Cortex_DFP", "cmsis-view", "cmsis-nn"):
            shutil.move(self.sdk / "core_software" / directory, self.root / directory)
        result = self.configure(
            f"""
set(CMSIS_PATH "{self.root}/cmsis_6")
set(CORTEX_DFP_PATH "{self.root}/Cortex_DFP")
set(CMSIS_VIEW_PATH "{self.root}/cmsis-view")
set(CMSIS_NN_LOCAL_PATH "{self.root}/cmsis-nn")
arm_ethos_u_default_fetch("${{sdk}}" fetch SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
if(fetch)
  message(FATAL_ERROR "External dependency paths should not enable fetch")
endif()
arm_ensure_ethos_u_content("${{sdk}}" unused OFF SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())
        self.assertFalse((self.root / "patched.txt").exists())

    def test_runner_accepts_cmsis_5_without_cortex_dfp(self):
        shutil.move(
            self.sdk / "core_software/cmsis_6", self.sdk / "core_software/cmsis"
        )
        (self.sdk / "core_software/cmsis/Device/ARM").mkdir(parents=True)
        shutil.rmtree(self.sdk / "core_software/Cortex_DFP")
        result = self.configure(
            """
set(CMSIS_VER 5)
arm_ethos_u_default_fetch("${sdk}" fetch SYSTEM_CONFIG Ethos_U55_High_End_Embedded)
if(fetch)
  message(FATAL_ERROR "CMSIS 5 should not require CMSIS 6 or Cortex DFP")
endif()
arm_ensure_ethos_u_content("${sdk}" unused OFF SYSTEM_CONFIG Ethos_U55_High_End_Embedded)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())

    def test_incomplete_overrides_fail_before_fetching(self):
        for variable in (
            "ETHOSU_CORE_SOFTWARE_PATH",
            "CORE_DRIVER_PATH",
            "CMSIS_PATH",
            "CMSIS_VIEW_PATH",
            "CMSIS_NN_LOCAL_PATH",
            "CORTEX_DFP_PATH",
        ):
            override = self.root / variable
            override.mkdir()
            for fetch in ("ON", "OFF"):
                with self.subTest(variable=variable, fetch=fetch):
                    result = self.configure(
                        f"""
set({variable} "{override}")
arm_ensure_ethos_u_content("${{sdk}}" unused {fetch}
  SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
"""
                    )
                    self.assertNotEqual(result.returncode, 0)
                    diagnostic = " ".join(result.stderr.split())
                    self.assertIn("Cannot automatically repair", diagnostic)
                    self.assertIn(str(override), diagnostic)
                    self.assertFalse((self.sdk / "fetch-called").exists())
                    self.assertFalse((self.root / "patched.txt").exists())

    def test_incomplete_override_is_detected_with_missing_managed_content(self):
        (self.sdk / SDK_FILES[1]).unlink()
        result = self.configure(
            f"""
set(CMSIS_VIEW_PATH "{self.root}/external cmsis-view")
arm_ensure_ethos_u_content("${{sdk}}" unused ON
  SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
"""
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot automatically repair", result.stderr)
        self.assertIn("external cmsis-view", result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())
        self.assertFalse((self.root / "patched.txt").exists())

    def test_explicit_managed_paths_can_be_repaired(self):
        self.event_recorder.unlink()
        result = self.configure(
            """
set(ETHOSU_CORE_SOFTWARE_PATH "${sdk}/core_software")
set(CORE_DRIVER_PATH "${sdk}/core_software/core_driver")
set(CMSIS_PATH "${sdk}/core_software/cmsis_6")
set(CMSIS_VIEW_PATH "${sdk}/core_software/./cmsis-view")
set(CMSIS_NN_LOCAL_PATH "${sdk}/core_software/cmsis-nn")
set(CORTEX_DFP_PATH "${sdk}/core_software/Cortex_DFP")
arm_ensure_ethos_u_content("${sdk}" unused ON
  SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
""",
            f"Path({json.dumps(SDK_FILES[-1])}).touch()\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())
        self.assertTrue(self.event_recorder.exists())

    def test_complete_override_allows_managed_content_repair(self):
        override = self.root / "external cmsis-view"
        shutil.copytree(self.sdk / "core_software/cmsis-view", override)
        self.event_recorder.unlink()
        (self.sdk / SDK_FILES[1]).unlink()
        result = self.configure(
            f"""
set(CMSIS_VIEW_PATH "{override}")
arm_ensure_ethos_u_content("${{sdk}}" unused ON
  SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)
""",
            f"Path({json.dumps(SDK_FILES[1])}).touch()\n"
            f"Path({json.dumps(SDK_FILES[-1])}).touch()\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())
        self.assertTrue((self.sdk / SDK_FILES[1]).exists())

    def test_incomplete_cmsis_5_fails_before_fetching(self):
        result = self.configure(
            """
set(CMSIS_VER 5)
arm_ensure_ethos_u_content("${sdk}" unused ON
  SYSTEM_CONFIG Ethos_U55_High_End_Embedded)
"""
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot automatically repair", result.stderr)
        self.assertIn("core_software/cmsis/", result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())
        self.assertFalse((self.root / "patched.txt").exists())

    def test_backend_leaves_missing_runner_dependencies_fetchable(self):
        self.event_recorder.unlink()
        result = self.configure(
            f"""
set(EXECUTORCH_ROOT "{ARM_ROOT.parents[1]}")
set(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
set(CMAKE_SKIP_INSTALL_RULES ON)
add_library(executorch_core INTERFACE)
add_library(ethosu_core_driver INTERFACE)
add_subdirectory("{ARM_ROOT}" arm_backend)
get_target_property(backend_includes executorch_delegate_ethos_u INCLUDE_DIRECTORIES)
list(FIND backend_includes "${{sdk}}/core_software/core_driver/include" sdk_include)
if(sdk_include EQUAL -1)
  message(FATAL_ERROR "Backend must use the temporary SDK")
endif()
if(DEFINED FETCH_ETHOS_U_CONTENT)
  message(FATAL_ERROR "Backend should leave the fetch default to the runner")
endif()
file(WRITE "{self.root}/backend-configured" "")
include("{ARM_ROOT}/cmake/ArmRunnerUtils.cmake")
function(patch_ethos_u_repo REPO_PATH)
  file(APPEND "{self.root}/patched.txt" "${{REPO_PATH}}\n")
endfunction()
arm_runner_configure_ethos_u_platform(
  SDK_PATH "${{sdk}}" SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid
  MEMORY_MODE Dedicated_Sram_384KB
)
"""
        )
        self.assertTrue((self.root / "backend-configured").exists(), result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to fetch Ethos-U externals", result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())

    def test_runner_requires_cmsis_nn(self):
        (self.sdk / "core_software/cmsis-nn/Include/arm_nnfunctions.h").unlink()
        for system_config in (
            "Ethos_U55_High_End_Embedded",
            "Ethos_U65_High_End",
            "Ethos_U85_SYS_DRAM_Mid",
        ):
            with self.subTest(system_config=system_config):
                result = self.configure(
                    f"""
arm_ethos_u_default_fetch("${{sdk}}" fetch SYSTEM_CONFIG {system_config})
if(NOT fetch)
  message(FATAL_ERROR "Missing CMSIS-NN should enable fetching")
endif()
arm_ensure_ethos_u_content("${{sdk}}" unused OFF SYSTEM_CONFIG {system_config})
"""
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Missing or incomplete Ethos-U content", result.stderr)
                self.assertFalse((self.sdk / "fetch-called").exists())
                self.assertFalse((self.root / "patched.txt").exists())

    def test_runner_repairs_missing_cmsis_nn(self):
        header = self.sdk / "core_software/cmsis-nn/Include/arm_nnfunctions.h"
        for system_config in (
            "Ethos_U55_High_End_Embedded",
            "Ethos_U65_High_End",
            "Ethos_U85_SYS_DRAM_Mid",
        ):
            with self.subTest(system_config=system_config):
                header.unlink()
                result = self.configure(
                    f"""
set(CMSIS_NN_LOCAL_PATH "${{sdk}}/core_software/cmsis-nn")
arm_ethos_u_default_fetch("${{sdk}}" fetch SYSTEM_CONFIG {system_config})
if(NOT fetch)
  message(FATAL_ERROR "Missing CMSIS-NN should enable fetching")
endif()
arm_ensure_ethos_u_content("${{sdk}}" unused "${{fetch}}"
  SYSTEM_CONFIG {system_config})
""",
                    'Path("core_software/cmsis-nn/Include/arm_nnfunctions.h").touch()\n',
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((self.sdk / "fetch-called").exists())
                self.assertTrue(header.exists())

    def test_missing_selected_platform_enables_fetch(self):
        for system_config, platform in (
            ("Ethos_U55_High_End_Embedded", "corstone-300"),
            ("Ethos_U65_High_End", "corstone-300"),
            ("Ethos_U85_SYS_DRAM_Mid", "corstone-320"),
        ):
            with self.subTest(system_config=system_config):
                path = self.sdk / "core_platform/targets" / platform / "CMakeLists.txt"
                path.unlink()
                result = self.configure(
                    f"""
arm_ethos_u_default_fetch("${{sdk}}" fetch SYSTEM_CONFIG {system_config})
if(NOT fetch)
  message(FATAL_ERROR "Missing selected platform should enable fetch")
endif()
arm_ensure_ethos_u_content("${{sdk}}" unused OFF SYSTEM_CONFIG {system_config})
"""
                )
                path.touch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Missing or incomplete Ethos-U content", result.stderr)
                self.assertFalse((self.sdk / "fetch-called").exists())

    def test_incomplete_sdk_with_fetch_disabled_fails(self):
        self.event_recorder.unlink()
        result = self.configure('arm_ensure_ethos_u_content("${sdk}" unused OFF)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or incomplete Ethos-U content", result.stderr)
        self.assertIn("FETCH_ETHOS_U_CONTENT=ON", result.stderr)
        self.assertFalse((self.sdk / "fetch-called").exists())

    def test_failed_fetch_stops_before_patching_dependencies(self):
        self.event_recorder.unlink()
        result = self.configure('arm_ensure_ethos_u_content("${sdk}" unused ON)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to fetch Ethos-U externals", result.stderr)
        self.assertIn("(23)", result.stderr)
        self.assertEqual(
            (self.root / "patched.txt").read_text().splitlines(), [str(self.sdk)]
        )

    def test_successful_process_with_incomplete_sdk_fails(self):
        self.event_recorder.unlink()
        result = self.configure(
            'arm_ensure_ethos_u_content("${sdk}" unused ON)', "pass\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Ethos-U externals are incomplete", result.stderr)
        self.assertEqual(
            (self.root / "patched.txt").read_text().splitlines(), [str(self.sdk)]
        )

    def test_partial_sdk_is_fetched_even_with_existing_driver(self):
        self.event_recorder.unlink()
        result = self.configure(
            'arm_ensure_ethos_u_content("${sdk}" unused ON)',
            "import sys\n"
            'assert sys.argv[1:] == ["-c", "26.08.json", "fetch"]\n'
            f"Path({json.dumps(SDK_FILES[-1])}).touch()\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())
        self.assertTrue(self.event_recorder.exists())
        self.assertEqual(
            (self.root / "patched.txt").read_text().splitlines(),
            [
                str(self.sdk),
                str(self.sdk / "core_software"),
                str(self.sdk / "core_platform"),
            ],
        )

    def test_missing_cortex_dfp_device_tree_is_fetched(self):
        device_tree = self.sdk / "core_software/Cortex_DFP/Device"
        shutil.rmtree(device_tree)
        self.assertTrue(
            (self.sdk / "core_software/Cortex_DFP/ARM.Cortex_DFP.pdsc").exists()
        )
        result = self.configure(
            'arm_ensure_ethos_u_content("${sdk}" unused ON '
            "SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)",
            'Path("core_software/Cortex_DFP/Device").mkdir()\n',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())
        self.assertTrue(device_tree.is_dir())
        self.assertEqual(
            (self.root / "patched.txt").read_text().splitlines(),
            [
                str(self.sdk),
                str(self.sdk / "core_software"),
                str(self.sdk / "core_platform"),
            ],
        )

    def test_fetch_without_cortex_dfp_device_tree_stops_before_patching(self):
        shutil.rmtree(self.sdk / "core_software/Cortex_DFP/Device")
        result = self.configure(
            'arm_ensure_ethos_u_content("${sdk}" unused ON '
            "SYSTEM_CONFIG Ethos_U85_SYS_DRAM_Mid)",
            "pass\n",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Ethos-U externals are incomplete", result.stderr)
        self.assertTrue((self.sdk / "fetch-called").exists())
        self.assertEqual(
            (self.root / "patched.txt").read_text().splitlines(), [str(self.sdk)]
        )


@unittest.skipUnless(
    shutil.which("git") and shutil.which("bash"), "Git and Bash are required"
)
class TestEthosUSetupPatches(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ethos patches ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin"
        self.repo = self.root / "ethos-u"
        self.patches = self.root / "patches"
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "ExecuTorch test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "ExecuTorch test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.git("init", "--quiet", str(self.origin), cwd=self.root)
        (self.origin / "data.txt").write_text("base\n")
        self.git("add", "data.txt", cwd=self.origin)
        self.git("commit", "--quiet", "-m", "Base", cwd=self.origin)
        self.git("tag", "26.05.1", cwd=self.origin)
        self.git("clone", "--quiet", str(self.origin), str(self.repo), cwd=self.root)
        self.base_rev = self.git("rev-parse", "HEAD").stdout.strip()
        (self.origin / "data.txt").write_text("updated\n")
        self.git("commit", "--quiet", "-am", "Update", cwd=self.origin)
        self.remote_rev = self.git("rev-parse", "HEAD", cwd=self.origin).stdout.strip()

    def git(self, *args, cwd=None, check=True):
        return (
            subprocess.run(  # nosec B603 B607 - local Git fixtures and fixed arguments
                ["git", *args],
                cwd=cwd or self.repo,
                env=self.env,
                capture_output=True,
                text=True,
                check=check,
                timeout=30,
            )
        )

    def patch(self, revision):
        return subprocess.run(  # nosec B603 B607 - trusted setup function and local fixtures
            [
                "bash",
                "-c",
                'source "$1" && patch_repo "$2" "$3" "$4"',
                "test_patch_repo",
                str(ARM_ROOT / "scripts/utils.sh"),
                str(self.repo),
                revision,
                str(self.patches),
            ],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_local_commit_needs_no_fetch(self):
        self.git("remote", "set-url", "origin", str(self.root / "unavailable"))
        result = self.patch(self.base_rev)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.base_rev)

    def test_symbolic_and_short_refs_require_fetch(self):
        self.git("remote", "set-url", "origin", str(self.root / "unavailable"))
        (self.repo / "data.txt").write_text("local edit\n")
        for revision in ("26.05.1", self.base_rev[:12]):
            with self.subTest(revision=revision):
                result = self.patch(revision)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((self.repo / "data.txt").read_text(), "local edit\n")

    def test_symbolic_release_uses_fetched_commit(self):
        self.git("tag", "--force", "26.05.1", cwd=self.origin)
        self.git("branch", "26.05.1", self.base_rev)
        result = self.patch("26.05.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.remote_rev)
        self.assertEqual((self.repo / "data.txt").read_text(), "updated\n")

    def test_missing_commit_is_fetched(self):
        self.assertNotEqual(
            self.git("cat-file", "-e", self.remote_rev, check=False).returncode, 0
        )
        result = self.patch(self.remote_rev)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.remote_rev)

    def test_patch_failure_is_reported(self):
        patch_dir = self.patches / self.repo.name
        patch_dir.mkdir(parents=True)
        (patch_dir / "0001.patch").write_text("invalid patch\n")
        result = self.patch(self.base_rev)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / ".git/rebase-apply").exists())

    @unittest.skipUnless(CMAKE, "CMake is required")
    def test_release_patch_stacks(self):
        source_sdk = ARM_ROOT.parents[1] / "examples/arm/arm-scratch/ethos-u"
        releases = (
            ("", "5ae010a083c6fa9f3a4d1d71b90db5742c87cc08"),
            ("core_software", "b5ffdb34fd5ad8004231eeed647fbafe58760683"),
            ("core_platform", "cec1a0ae3f05b2cf9a1518c7087cda96aed322a0"),
        )
        for name, revision in releases:
            source = source_sdk / name
            if (
                not (source / ".git").exists()
                or self.git(
                    "cat-file", "-e", f"{revision}^{{commit}}", cwd=source, check=False
                ).returncode
            ):
                self.skipTest(
                    "Run examples/arm/setup.sh to provision the release repositories"
                )

        sdk = self.root / "real" / "ethos-u"
        sdk.parent.mkdir()
        for name, revision in releases:
            destination = sdk / name
            self.git(
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(source_sdk / name),
                str(destination),
                cwd=self.root,
            )
            self.git("checkout", "--quiet", revision, cwd=destination)
            self.git(
                "remote",
                "set-url",
                "origin",
                str(self.root / "unavailable"),
                cwd=destination,
            )

        # External dependencies need only readiness sentinels; Git patching stays real.
        for name in SDK_FILES:
            file = sdk / name
            file.parent.mkdir(parents=True, exist_ok=True)
            if not file.exists():
                file.touch()
        (sdk / "core_software/Cortex_DFP/Device").mkdir(parents=True, exist_ok=True)
        project = self.root / "cmake"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(
            f"""
cmake_minimum_required(VERSION 3.19)
project(ethos_patch_test NONE)
include("{ARM_ROOT}/scripts/corstone_utils.cmake")
fetch_ethos_u_content("{sdk}" "{ARM_ROOT.parents[1]}")
"""
        )
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                result = subprocess.run(  # nosec B603 - trusted CMake and local fixture project
                    [
                        CMAKE,
                        "-S",
                        str(project),
                        "-B",
                        str(self.root / "build"),
                        f"-DFETCHCONTENT_SOURCE_DIR_ETHOS_U={sdk}",
                        f"-DPython3_EXECUTABLE={sys.executable}",
                    ],
                    env=self.env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for name, revision in releases:
                    repository = sdk / name
                    patch_dir = (
                        ARM_ROOT.parents[1]
                        / "examples/arm/ethos-u-setup"
                        / repository.name
                    )
                    count = self.git(
                        "rev-list", "--count", f"{revision}..HEAD", cwd=repository
                    )
                    self.assertEqual(
                        int(count.stdout), len(list(patch_dir.glob("*.patch")))
                    )
                manifest = json.loads((sdk / "26.08.json").read_text())
                self.assertEqual(
                    {entry["path"] for entry in manifest["externals"]},
                    {
                        "core_platform",
                        "core_software",
                        "core_software/core_driver",
                        "core_software/cmsis_6",
                        "core_software/Cortex_DFP",
                        "core_software/cmsis-nn",
                        "core_software/cmsis-view",
                    },
                )


if __name__ == "__main__":
    unittest.main()
