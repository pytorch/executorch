# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / ".ci/scripts/install_pinned_pytorch.sh"
PIN = "a" * 40

# Stand-ins exercise the shell flow without downloading or building PyTorch.
TOOL = r"""#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ['TEST_ROOT'])
with (root / 'commands').open('a') as out:
    out.write(json.dumps([name, *args]) + '\n')
if name == 'uname':
    print('arm64' if args else os.environ.get('TEST_SYSTEM', 'Linux'))
elif name == 'install_name_tool':
    if args[0] == '-delete_rpath':
        marker = root / 'rpath-removed'
        if marker.exists():
            sys.exit(1)
        marker.touch()
elif name == 'git':
    if args[0] == 'clone':
        path = pathlib.Path(args[-1])
        path.mkdir()
        (path / 'version.txt').write_text('2.15.0a0')
    elif args[0] == 'ls-remote':
        refs = os.environ.get('TEST_REMOTE_REFS', 'a' * 40 + '\trefs/heads/release/2.15')
        print(refs)
        if not refs:
            sys.exit(2)
    elif args[0] == 'rev-parse':
        print('a' * (7 if '--short=7' in args else 40))
elif name in ('chosen-python', 'python'):
    if args[:2] == ['-m', 'venv']:
        directory = pathlib.Path(args[-1]) / 'bin'
        directory.mkdir(parents=True)
        (directory / 'python').symlink_to(root / 'bin' / 'chosen-python')
    elif args[:2] == ['-m', 'build']:
        if os.environ.get('TEST_BUILD_FAIL'):
            print('Build failed; installed pin: ' + (root / 'installed').read_text(), file=sys.stderr)
            sys.exit(42)
        platform = os.environ.get('TEST_WHEEL_PLATFORM') or ('macosx_15_0_arm64' if os.environ.get('TEST_SYSTEM') == 'Darwin' else 'linux_x86_64')
        pathlib.Path('dist').mkdir(exist_ok=True)
        pathlib.Path(f'dist/torch-2.15.0a0+gitaaaaaaa-cp312-cp312-{platform}.whl').touch()
    elif args[:3] == ['-m', 'pip', 'download']:
        if not os.environ.get('TEST_CACHE_HIT'):
            sys.exit(1)
        directory = pathlib.Path(args[args.index('--dest') + 1])
        directory.mkdir(exist_ok=True)
        (directory / args[-1].rsplit('/', 1)[1].replace('%2B', '+')).touch()
    elif args[:3] == ['-m', 'pip', 'install']:
        if any(arg.endswith('.whl') for arg in args):
            (root / 'installed').write_text(os.environ.get('TEST_BUILT_PIN', 'a' * 40))
    elif args[:3] == ['-m', 'pip', 'uninstall']:
        (root / 'installed').write_text('')
    elif args[0] == '-c':
        code = args[1]
        if 'sys.executable' in code:
            print(sys.argv[0])
        elif 'git_version' in code:
            print((root / 'installed').read_text())
        elif 'mac_ver' in code:
            print('macosx_15_0_arm64')
        elif 'python_version_tuple' in code:
            print('312')
        elif 'has_lapack' in code and os.environ.get('TEST_NO_LAPACK'):
            sys.exit('PyTorch was built without LAPACK')
        elif 'find_spec' in code:
            print(root / 'torch')
"""


class TestPinnedPytorch(unittest.TestCase):
    def run_installer(self, installed="", ref=PIN, verify=False, system="Linux", **env):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "installed").write_text(installed)
            (root / "torch/lib").mkdir(parents=True)
            (root / "torch/lib/libtorch.dylib").touch()
            for name in (
                "git",
                "uname",
                "chosen-python",
                "python",
                "aws",
                "install_name_tool",
            ):
                tool = root / "bin" / name
                tool.write_text(TOOL)
                tool.chmod(0o755)
            pin = root / ".ci/docker/ci_commit_pins/pytorch.txt"
            pin.parent.mkdir(parents=True)
            pin.write_text(ref)
            result = subprocess.run(
                [
                    "bash",
                    str(INSTALLER),
                    *(["--verify-only"] if verify else []),
                    str(root),
                ],
                env={
                    **os.environ,
                    "PATH": f"{root}/bin:" + os.environ["PATH"],
                    "PYTHON_EXECUTABLE": str(root / "bin/chosen-python"),
                    "TMPDIR": str(root),
                    "TEST_ROOT": str(root),
                    "TEST_SYSTEM": system,
                    "GITHUB_RUNNER": "1",
                    **env,
                },
                capture_output=True,
                text=True,
            )
            commands = root / "commands"
            calls = (
                [json.loads(line) for line in commands.read_text().splitlines()]
                if commands.exists()
                else []
            )
            self.assertEqual(list(root.glob("executorch-pytorch.*")), [])
            return result, calls

    def test_matching_pin_installs_domains_without_rebuilding_torch(self):
        for ref in (PIN, "release/2.15"):
            with self.subTest(ref=ref):
                result, calls = self.run_installer(
                    installed=PIN, ref=ref, system="Darwin"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(call[:2] == ["git", "clone"] for call in calls))
                self.assertFalse(any(call[1:3] == ["-m", "build"] for call in calls))
                self.assertFalse(any("uninstall" in call for call in calls))
                for package in ("audio", "vision"):
                    self.assertTrue(
                        any(
                            "--no-deps" in call
                            and any(f"{package}.git@" in arg for arg in call)
                            for call in calls
                        )
                    )

    def test_matching_macos_pin_repairs_rpaths_only_in_install_mode(self):
        for verify in (False, True):
            with self.subTest(verify=verify):
                result, calls = self.run_installer(
                    installed=PIN, system="Darwin", verify=verify
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                repairs = [call[1] for call in calls if call[0] == "install_name_tool"]
                self.assertEqual(
                    repairs,
                    [] if verify else ["-delete_rpath", "-delete_rpath", "-add_rpath"],
                )

    def test_verify_annotated_tag_uses_commit(self):
        result, _ = self.run_installer(
            installed=PIN,
            ref="v2.15.0",
            verify=True,
            TEST_REMOTE_REFS="b" * 40
            + "\trefs/tags/v2.15.0\n"
            + PIN
            + "\trefs/tags/v2.15.0^{}",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_remote_ref_fails_without_installing(self):
        for refs in (
            "",
            f"{PIN}\trefs/heads/release/2.15\n{'b' * 40}\trefs/tags/release/2.15",
        ):
            with self.subTest(refs=refs):
                result, calls = self.run_installer(
                    ref="release/2.15", TEST_REMOTE_REFS=refs
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot resolve PyTorch ref", result.stderr)
                self.assertFalse(any(call[1:3] == ["-m", "pip"] for call in calls))

    def test_verify_matching_pin_does_not_clone_or_install(self):
        for ref in (PIN, "release/2.15"):
            with self.subTest(ref=ref):
                result, calls = self.run_installer(installed=PIN, ref=ref, verify=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(call[:2] == ["git", "clone"] for call in calls))
                self.assertFalse(any(call[1:3] == ["-m", "pip"] for call in calls))
                self.assertTrue(
                    any("has_lapack" in arg for call in calls for arg in call)
                )

    def test_verify_mismatch_does_not_install(self):
        result, calls = self.run_installer(installed="b" * 40, verify=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pin mismatch", result.stderr)
        self.assertFalse(
            any(
                call[:2] == ["git", "clone"] or call[1:3] == ["-m", "pip"]
                for call in calls
            )
        )

    def test_linux_replaces_installed_build_in_both_directions(self):
        for installed in ("older", "newer"):
            with self.subTest(installed=installed):
                result, calls = self.run_installer(installed=installed)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(
                    any(call[1:4] == ["-m", "pip", "uninstall"] for call in calls)
                )
                self.assertTrue(any(call[1:3] == ["-m", "build"] for call in calls))
                self.assertFalse(any("--user" in call for call in calls))

    def test_build_failure_preserves_installed_torch(self):
        for system in ("Linux", "Darwin"):
            with self.subTest(system=system):
                result, calls = self.run_installer(
                    installed="old-pin", system=system, TEST_BUILD_FAIL="1"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.returncode, 42)
                self.assertIn("Build failed; installed pin: old-pin", result.stderr)
                self.assertIn(
                    "Failed to prepare the replacement PyTorch wheel", result.stderr
                )
                self.assertIn("have not been removed", result.stderr)
                self.assertFalse(any("uninstall" in call for call in calls))

    def test_branch_ref_resolves_before_matching(self):
        result, calls = self.run_installer(installed=PIN, ref="release/2.15")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(call[:2] == ["git", "ls-remote"] for call in calls))
        self.assertFalse(any(call[1:3] == ["-m", "build"] for call in calls))

    def test_wrong_built_commit_fails_verification(self):
        result, _ = self.run_installer(TEST_BUILT_PIN="b" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pin mismatch", result.stderr)
        self.assertNotIn("have not been removed", result.stderr)

    def test_macos_uses_cached_wheel(self):
        result, calls = self.run_installer(system="Darwin", TEST_CACHE_HIT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call[1:3] == ["-m", "build"] for call in calls))

    def test_macos_cache_miss_builds_matching_wheel(self):
        result, calls = self.run_installer(system="Darwin")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(call[1:3] == ["-m", "build"] for call in calls))

    def test_macos_upload_requires_github_actions(self):
        for actions, runner, upload in (
            ("", "", False),
            ("false", "", False),
            ("true", "", True),
            ("true", "1", False),
        ):
            with self.subTest(actions=actions, runner=runner):
                result, calls = self.run_installer(
                    system="Darwin", GITHUB_ACTIONS=actions, GITHUB_RUNNER=runner
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(any(call[0] == "aws" for call in calls), upload)

    def test_macos_rejects_wrong_cache_filename(self):
        result, calls = self.run_installer(
            installed="old-pin", system="Darwin", TEST_WHEEL_PLATFORM="wrong"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match cache key", result.stderr)
        self.assertFalse(any("uninstall" in call for call in calls))

    def test_verify_branch_mismatch_does_not_install(self):
        result, calls = self.run_installer(
            installed="b" * 40, ref="release/2.15", verify=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pin mismatch", result.stderr)
        self.assertFalse(any(call[1:3] == ["-m", "pip"] for call in calls))

    def test_python_from_path_is_supported(self):
        result, _ = self.run_installer(installed=PIN, PYTHON_EXECUTABLE="python")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_matching_pin_still_checks_lapack(self):
        result, _ = self.run_installer(installed=PIN, TEST_NO_LAPACK="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without LAPACK", result.stderr)

    def test_docker_wrapper_preserves_conda_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(ROOT / ".ci/docker/common/install_pytorch.sh", root)
            (root / "utils.sh").write_text(
                "conda_run() {\n"
                '  test "$TORCH_VERSION" = release/2.15 &&\n'
                '  test "$PYTHON_EXECUTABLE" = /opt/conda/envs/py_3.12/bin/python &&\n'
                '  test "$1" = bash &&\n'
                '  test "$2" = "' + directory + '/install_pinned_pytorch.sh"\n'
                "}\n"
            )
            result = subprocess.run(
                ["bash", str(root / "install_pytorch.sh")],
                env={
                    **os.environ,
                    "TORCH_VERSION": "release/2.15",
                    "PYTHON_VERSION": "3.12",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_windows_is_rejected(self):
        result, _ = self.run_installer(system="MINGW64_NT")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported system", result.stderr)


if __name__ == "__main__":
    unittest.main()
