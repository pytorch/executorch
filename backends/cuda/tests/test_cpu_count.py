# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from executorch.backends.cuda.utils import cpu_count


class AvailableCpuCountTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.mount = Path(self.temporary.name)
        self.proc_cgroup = "0::/job\n"
        self.proc_mountinfo = f"1 0 0:1 / {self.mount} rw - cgroup2 cgroup rw\n"
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == Path("/proc/self/cgroup"):
                return self.proc_cgroup
            if path == Path("/proc/self/mountinfo"):
                return self.proc_mountinfo
            return original_read(path, *args, **kwargs)

        for patch in (
            mock.patch.object(cpu_count.sys, "platform", "linux"),
            mock.patch.object(cpu_count.os, "cpu_count", return_value=44),
            mock.patch.object(
                cpu_count.os,
                "sched_getaffinity",
                return_value=set(range(44)),
                create=True,
            ),
            mock.patch.object(Path, "read_text", read_text),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def quota(self, group, quota, period=100000, unified=True):
        directory = self.mount / group
        directory.mkdir(parents=True, exist_ok=True)
        if unified:
            (directory / "cpu.max").write_text(f"{quota} {period}\n")
        else:
            (directory / "cpu.cfs_quota_us").write_text(str(quota))
            (directory / "cpu.cfs_period_us").write_text(str(period))

    def test_unlimited_host_preserves_cpu_count(self):
        self.quota("job", "max")
        self.assertEqual(cpu_count.available_cpu_count(), 44)

    def test_affinity_restricts_available_cpus(self):
        self.quota("job", "max")
        with mock.patch.object(
            cpu_count.os, "sched_getaffinity", return_value=set(range(7))
        ):
            self.assertEqual(cpu_count.available_cpu_count(), 7)

    def test_cgroup_v2_quota_without_affinity_limit(self):
        self.quota("job", 1100000)
        self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_tightest_visible_parent_quota(self):
        self.proc_cgroup = "0::/parent/job\n"
        self.quota("parent/job", "max")
        self.quota("parent", 1100000)
        self.quota("", 800000)
        self.assertEqual(cpu_count.available_cpu_count(), 8)

    def test_mount_root_maps_host_cgroup_path(self):
        self.proc_cgroup = "0::/containers/container/job\n"
        self.proc_mountinfo = (
            f"1 0 0:1 /containers/container {self.mount} rw - cgroup2 cgroup rw\n"
        )
        self.quota("job", 1100000)
        self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_cgroup_namespace_root_maps_mountpoint(self):
        self.proc_cgroup = "0::/\n"
        self.proc_mountinfo = (
            f"1 0 0:1 /containers/container {self.mount} rw - cgroup2 cgroup rw\n"
        )
        self.quota("", 1100000)
        self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_cgroup_v1_combined_controllers(self):
        self.proc_cgroup = "4:cpu,cpuacct:/job\n2:memory:/other\n"
        self.proc_mountinfo = (
            f"1 0 0:1 / {self.mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
        )
        self.quota("job", 1100000, unified=False)
        self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_cgroup_v1_unlimited_parent_can_restrict(self):
        self.proc_cgroup = "4:cpu:/parent/job\n"
        self.proc_mountinfo = f"1 0 0:1 / {self.mount} rw - cgroup cgroup rw,cpu\n"
        self.quota("parent/job", -1, unified=False)
        self.quota("parent", 1100000, unified=False)
        self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_fractional_quota_rounds_up_to_a_worker(self):
        for quota, expected in ((50000, 1), (150000, 2)):
            with self.subTest(quota=quota):
                self.quota("job", quota)
                self.assertEqual(cpu_count.available_cpu_count(), expected)

    def test_affinity_tighter_than_quota(self):
        self.quota("job", 1100000)
        with mock.patch.object(cpu_count.os, "sched_getaffinity", return_value={0, 1}):
            self.assertEqual(cpu_count.available_cpu_count(), 2)

    def test_missing_proc_files_falls_back(self):
        with mock.patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertEqual(cpu_count.available_cpu_count(), 44)

    def test_invalid_or_missing_quota_does_not_hide_parent(self):
        self.proc_cgroup = "0::/parent/job\n"
        self.quota("parent", 1100000)
        for value in (None, "invalid", "0 100000", "1100000 0"):
            with self.subTest(value=value):
                if value is not None:
                    self.quota("parent/job", 1)
                    (self.mount / "parent/job/cpu.max").write_text(value)
                self.assertEqual(cpu_count.available_cpu_count(), 11)

    def test_non_linux_without_affinity(self):
        with (
            mock.patch.object(cpu_count.sys, "platform", "win32"),
            mock.patch.object(
                cpu_count.os, "sched_getaffinity", side_effect=AttributeError
            ),
        ):
            self.assertEqual(cpu_count.available_cpu_count(), 44)

    def test_missing_cpu_count_still_returns_one(self):
        with mock.patch.object(cpu_count.os, "cpu_count", return_value=None):
            self.assertEqual(cpu_count.available_cpu_count(), 1)


if __name__ == "__main__":
    unittest.main()
