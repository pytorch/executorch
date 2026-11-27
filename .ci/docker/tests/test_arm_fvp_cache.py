# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_HELPER = REPO_ROOT / ".ci/docker/common/arm_fvp_cache.sh"
FVP_UTILS = REPO_ROOT / "backends/arm/scripts/fvp_utils.sh"
SETUP = REPO_ROOT / "examples/arm/setup.sh"
FAKE_METADATA = """
arm_fvp_select_platform() {
    ARM_FVP_URL="https://invalid.test/${2}_${1}_${FVP_TEST_VERSION:-1}.tgz"
    ARM_FVP_MD5="${FVP_TEST_MD5}"
    ARM_FVP_MODEL_DIR=Linux64_GCC-9.3
}
"""


@unittest.skipUnless(shutil.which("flock"), "Cache population requires Linux flock")
class ArmFvpCacheTest(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.cache = self.root / "shared cache"
        self.payload = self.root / "payload"
        self.payload.write_bytes(b"fake FVP archive")
        self.calls = self.root / "downloads"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        curl = bin_dir / "curl"
        curl.write_text(
            f"#!{sys.executable}\n"
            "import os, shutil, sys, time\n"
            "from pathlib import Path\n"
            "with open(os.environ['FVP_TEST_CALLS'], 'a') as calls:\n"
            "    calls.write(' '.join(sys.argv[1:]) + '\\n')\n"
            "time.sleep(float(os.environ.get('FVP_TEST_DELAY', '0')))\n"
            "output = sys.argv[sys.argv.index('--output') + 1]\n"
            "shutil.copyfile(os.environ['FVP_TEST_PAYLOAD'], output)\n"
            "sys.exit(int(os.environ.get('FVP_TEST_CURL_STATUS', '0')))\n"
        )
        curl.chmod(0o755)
        sleep = bin_dir / "sleep"
        sleep.write_text("#!/bin/sh\nexit 0\n")
        sleep.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            PATH=f"{bin_dir}:{self.env['PATH']}",
            ARM_FVP_CACHE_DIR=str(self.cache),
            ARM_FVP_CACHE_ONLY="0",
            ARM_SETUP_DOWNLOAD_CACHE="",
            FVP_TEST_CALLS=str(self.calls),
            FVP_TEST_PAYLOAD=str(self.payload),
            FVP_TEST_MD5=hashlib.md5(
                self.payload.read_bytes(), usedforsecurity=False
            ).hexdigest(),
            FVP_TEST_HELPER=str(CACHE_HELPER),
            FVP_TEST_UTILS=str(FVP_UTILS),
        )

    def cache_command(self, arch="x86_64", platform="corstone300", legacy=""):
        return [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'source "$FVP_TEST_HELPER"\n'
            + FAKE_METADATA
            + 'arm_fvp_cache_archive "$1" "$2" "$3"\n'
            + 'printf "%s\\n" "$ARM_FVP_ARCHIVE"\n',
            "cache-test",
            arch,
            platform,
            str(legacy),
        ]

    def run_cache(self, *args, env=None, cwd=None):
        return subprocess.run(
            self.cache_command(*args),
            env=env or self.env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def download_count(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def test_fresh_workspaces_reuse_one_download(self):
        archives = []
        for name in ("first-job", "second-job"):
            workspace = self.root / name
            workspace.mkdir()
            result = self.run_cache(cwd=workspace)
            self.assertEqual(result.returncode, 0, result.stderr)
            archives.append(result.stdout.strip())
        self.assertEqual(archives[0], archives[1])
        self.assertEqual(Path(archives[0]).read_bytes(), self.payload.read_bytes())
        self.assertEqual(self.download_count(), 1)
        self.assertIn("ExecuTorch-Arm-FVP-Setup/1", self.calls.read_text())

    def test_legacy_archive_is_preserved_and_migrated(self):
        legacy = self.root / "FVP_corstone300.tgz"
        shutil.copyfile(self.payload, legacy)
        result = self.run_cache("x86_64", "corstone300", legacy)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(legacy.read_bytes(), self.payload.read_bytes())
        self.assertEqual(Path(result.stdout.strip()).parent, self.cache)
        self.assertEqual(self.download_count(), 0)

    def test_cache_only_miss_never_downloads(self):
        self.env["ARM_FVP_CACHE_ONLY"] = "1"
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbids downloads", result.stderr)
        self.assertEqual(self.download_count(), 0)
        self.assertFalse(self.cache.exists())

    def test_cache_only_accepts_valid_legacy_archive(self):
        legacy = self.root / "FVP_corstone300.tgz"
        shutil.copyfile(self.payload, legacy)
        self.env["ARM_FVP_CACHE_ONLY"] = "1"
        result = self.run_cache("x86_64", "corstone300", legacy)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(legacy))
        self.assertEqual(self.download_count(), 0)
        self.assertFalse(self.cache.exists())

    def test_setup_download_cache_migrates_after_invalid_scratch(self):
        downloads = self.root / "old download cache"
        downloads.mkdir()
        archive = downloads / "corstone300_x86_64_1.tgz"
        shutil.copyfile(self.payload, archive)
        archive.chmod(0o444)
        downloads.chmod(0o555)
        self.addCleanup(downloads.chmod, 0o755)
        self.env["ARM_SETUP_DOWNLOAD_CACHE"] = str(downloads)
        scratch = self.root / "FVP_corstone300.tgz"
        scratch.write_bytes(b"truncated")

        result = self.run_cache("x86_64", "corstone300", scratch)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()).parent, self.cache)
        self.assertEqual(
            Path(result.stdout.strip()).read_bytes(), self.payload.read_bytes()
        )
        self.assertEqual(archive.read_bytes(), self.payload.read_bytes())
        self.assertEqual(scratch.read_bytes(), b"truncated")
        self.assertEqual(self.download_count(), 0)
        self.assertEqual(list(self.cache.glob("*.tmp.*")), [])

    def test_cache_only_reuses_relative_setup_download_cache_without_writes(self):
        downloads = self.root / "old download cache"
        downloads.mkdir()
        archive = downloads / "corstone300_x86_64_1.tgz"
        shutil.copyfile(self.payload, archive)
        archive.chmod(0o444)
        downloads.chmod(0o555)
        self.addCleanup(downloads.chmod, 0o755)
        self.env["ARM_SETUP_DOWNLOAD_CACHE"] = downloads.name
        self.env["ARM_FVP_CACHE_ONLY"] = "1"

        result = self.run_cache(cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = Path(result.stdout.strip())
        self.assertTrue(selected.is_absolute())
        self.assertEqual(selected.resolve(), archive.resolve())
        self.assertEqual(archive.read_bytes(), self.payload.read_bytes())
        self.assertEqual(list(downloads.iterdir()), [archive])
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.download_count(), 0)

    def test_corrupt_setup_download_cache_is_not_reused_or_modified(self):
        downloads = self.root / "old download cache"
        downloads.mkdir()
        archive = downloads / "corstone300_x86_64_1.tgz"
        archive.write_bytes(b"truncated")
        self.env["ARM_SETUP_DOWNLOAD_CACHE"] = str(downloads)
        self.env["ARM_FVP_CACHE_ONLY"] = "1"
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbids downloads", result.stderr)
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.download_count(), 0)

        self.env["ARM_FVP_CACHE_ONLY"] = "0"
        result = self.run_cache()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.strip()).read_bytes(), self.payload.read_bytes()
        )
        self.assertEqual(archive.read_bytes(), b"truncated")
        self.assertEqual(self.download_count(), 1)

    def test_corrupt_cache_is_rejected_then_repaired(self):
        initial = self.run_cache()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        archive = Path(initial.stdout.strip())
        archive.write_bytes(b"truncated")
        self.env["ARM_FVP_CACHE_ONLY"] = "1"
        rejected = self.run_cache()
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(self.download_count(), 1)
        self.env["ARM_FVP_CACHE_ONLY"] = "0"
        repaired = self.run_cache()
        self.assertEqual(repaired.returncode, 0, repaired.stderr)
        self.assertEqual(archive.read_bytes(), self.payload.read_bytes())
        self.assertEqual(self.download_count(), 2)

    def test_failed_downloads_leave_no_archive_or_temporary_file(self):
        self.env["FVP_TEST_CURL_STATUS"] = "22"
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.download_count(), 5)
        self.assertEqual(list(self.cache.glob("*.tgz")), [])
        self.assertEqual(list(self.cache.glob("*.tmp.*")), [])

    def test_wrong_checksum_is_not_published(self):
        self.payload.write_bytes(b"unexpected response body")
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.download_count(), 5)
        self.assertEqual(list(self.cache.glob("*.tgz")), [])

    def test_concurrent_jobs_share_one_complete_download(self):
        self.env["FVP_TEST_DELAY"] = "0.2"
        processes = [
            subprocess.Popen(
                self.cache_command(),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            outputs.append(stdout.strip())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(self.download_count(), 1)
        self.assertEqual(Path(outputs[0]).read_bytes(), self.payload.read_bytes())

    def test_architecture_and_version_have_separate_entries(self):
        paths = []
        for arch, version in (("x86_64", "1"), ("aarch64", "1"), ("x86_64", "2")):
            self.env["FVP_TEST_VERSION"] = version
            result = self.run_cache(arch)
            self.assertEqual(result.returncode, 0, result.stderr)
            paths.append(result.stdout.strip())
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(self.download_count(), 3)

    def test_read_only_hit_needs_no_lock_or_write(self):
        self.cache.mkdir()
        archive = self.cache / f"{self.env['FVP_TEST_MD5']}-corstone300_x86_64_1.tgz"
        shutil.copyfile(self.payload, archive)
        archive.chmod(0o444)
        self.cache.chmod(0o555)
        self.addCleanup(self.cache.chmod, 0o755)
        self.env["ARM_FVP_CACHE_ONLY"] = "1"
        result = self.run_cache()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(archive))
        self.assertEqual(list(self.cache.iterdir()), [archive])
        self.assertEqual(self.download_count(), 0)

    def test_selected_platform_install_preserves_legacy_archive(self):
        with tarfile.open(self.payload, "w:gz") as archive:
            installer = b"#!/bin/sh\nmkdir -p models/Linux64_GCC-9.3\ntouch installed\n"
            info = tarfile.TarInfo("FVP_Corstone_SSE-300.sh")
            info.size = len(installer)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(installer))
        self.env["FVP_TEST_MD5"] = hashlib.md5(
            self.payload.read_bytes(), usedforsecurity=False
        ).hexdigest()
        scratch = self.root / "scratch"
        scratch.mkdir()
        legacy = scratch / "FVP_corstone300.tgz"
        shutil.copyfile(self.payload, legacy)
        result = subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                'ARCH=x86_64; OS=Linux; root_dir="$1"; fvps=(corstone300)\n'
                'source "$FVP_TEST_UTILS"\n'
                + FAKE_METADATA
                + 'install_fvp\nsetup_path_script="$root_dir/setup_path"\nsetup_path_fvp\n',
                "install-test",
                str(scratch),
            ],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((scratch / "FVP-corstone300/installed").exists())
        self.assertFalse((scratch / "FVP-corstone320").exists())
        self.assertEqual(legacy.read_bytes(), self.payload.read_bytes())
        setup_path = (scratch / "setup_path.sh").read_text()
        self.assertIn("FVP-corstone300/python/lib/", setup_path)
        self.assertNotIn("FVP-corstone320/", setup_path)
        self.assertEqual(self.download_count(), 0)


class ArmFvpMacTest(unittest.TestCase):
    def test_macos_keeps_linux_cache_untouched(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            wrapper = bin_dir / "FVP_Corstone_SSE-300_Ethos-U55"
            wrapper.write_text("#!/bin/sh\nexit 0\n")
            wrapper.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=f"{bin_dir}:/usr/bin:/bin",
                FVP_TEST_UTILS=str(FVP_UTILS),
                ARM_FVP_CACHE_DIR=str(root / "cache"),
                ARM_FVP_CACHE_ONLY="1",
            )
            result = subprocess.run(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    "ARCH=arm64; OS=Darwin; fvps=(corstone300 corstone320)\n"
                    'source "$FVP_TEST_UTILS"\n'
                    "setup_fvp\n"
                    "check_fvp_eula\n"
                    "install_fvp\n"
                    "setup_path_fvp\n",
                    "mac-setup-test",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "cache").exists())


class ArmFvpOptionsTest(unittest.TestCase):
    def test_missing_or_unknown_platform_fails_before_setup(self):
        for args in (("--fvp",), ("--fvp", "unknown")):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(SETUP), *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fvp", (result.stdout + result.stderr).lower())

    def test_repeatable_fvp_option_deduplicates_selection(self):
        result = subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                'source "$1"\n'
                "check_options --fvp corstone320 --fvp corstone320 --fvp corstone1000\n"
                'printf "%s\\n" "${fvps[*]}"\n',
                "fvp-options-test",
                str(SETUP),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "corstone320 corstone1000")

    def test_setup_defaults_to_corstone300_and_corstone320(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ("pip", "FVP_Corstone_SSE-300_Ethos-U55"):
                wrapper = bin_dir / name
                wrapper.write_text("#!/bin/sh\nexit 0\n")
                wrapper.chmod(0o755)
            uname = bin_dir / "uname"
            uname.write_text(
                '#!/bin/sh\nif [ "$1" = "-s" ]; then echo Darwin; else echo arm64; fi\n'
            )
            uname.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=f"{bin_dir}:/usr/bin:/bin",
                ARM_FVP_CACHE_DIR=str(root / "cache"),
                ARM_FVP_CACHE_ONLY="1",
            )
            result = subprocess.run(
                [
                    "bash",
                    str(SETUP),
                    "--root-dir",
                    str(root / "scratch"),
                    "--disable-ethos-u-deps",
                    "--enable-fvps",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "Setting up Arm Fixed Virtual Platforms: corstone300,corstone320",
                result.stdout,
            )
            self.assertFalse((root / "cache").exists())


if __name__ == "__main__":
    unittest.main()
