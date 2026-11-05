# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the wheel platform tag checks.

Here rather than in the wheel checks, because nothing under test needs a built wheel.
Running it as part of the wheel checks meant it needed eleven built wheels to exercise
one comparison, and it still could not run on a machine that had not built one.

The tag comparison had a real defect that this covers. The release pipeline builds in a
manylinux image and rewrites the wheel's file name, so the tag on the file and the tag
auditwheel reports never agree in spelling, and comparing them as text rejected every
correct wheel. No local build reproduces that rewrite, so nothing short of a unit test
catches it before CI.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wheel"))

import test_shared_libraries  # noqa: E402
from test_shared_libraries import (  # noqa: E402
    _mach_o_minimum,
    _macos_minimum_violations,
    _metallib_minimum,
    _tag_architectures_match,
    _wheel_architecture,
)


@pytest.mark.parametrize(
    "claimed,supported",
    [
        # What the release pipeline actually produces: it builds in a manylinux image
        # and rewrites the file name, while auditwheel reports a plain linux tag
        # because the wheel depends on torch without vendoring torch's libraries.
        ("manylinux_2_28_x86_64", "linux_x86_64"),
        ("manylinux_2_28_aarch64", "linux_aarch64"),
        # The legacy spelling, which has no underscore before its version.
        ("manylinux2014_x86_64", "linux_x86_64"),
        ("manylinux2014_aarch64", "linux_aarch64"),
        # A local build, where nothing rewrites the name.
        ("linux_x86_64", "linux_x86_64"),
        ("linux_aarch64", "linux_aarch64"),
    ],
)
def test_accepts_tags_the_release_pipeline_produces(claimed, supported):
    assert _tag_architectures_match(claimed, supported) is True


@pytest.mark.parametrize(
    "claimed,supported",
    [
        ("manylinux_2_28_aarch64", "linux_x86_64"),
        ("manylinux_2_28_x86_64", "linux_aarch64"),
        ("linux_aarch64", "linux_x86_64"),
    ],
)
def test_rejects_an_architecture_mismatch(claimed, supported):
    """A wheel labelled for the wrong architecture installs where it cannot run."""
    assert _tag_architectures_match(claimed, supported) is False


@pytest.mark.parametrize("tag", ["win_amd64", "macosx_11_0_arm64", "any", "", "linux"])
def test_reports_a_tag_it_cannot_read(tag):
    """None, not False, so an unreadable tag is not mistaken for a mismatch."""
    assert _wheel_architecture(tag) is None
    assert _tag_architectures_match(tag, "linux_x86_64") is None


def test_reads_every_architecture_the_project_builds_for():
    for architecture in ("x86_64", "aarch64", "i686", "ppc64le", "s390x", "armv7l"):
        assert _wheel_architecture(f"linux_{architecture}") == architecture
        assert _wheel_architecture(f"manylinux_2_28_{architecture}") == architecture


@pytest.mark.parametrize(
    "tag,minimums,violations",
    [
        # Binaries at or below the supported minimum.
        ("macosx_14_0_arm64", {"a.dylib": (14, 0), "b.so": (13, 0)}, (False, [])),
        # A newer major version, after bdist_wheel raised the tag to match it.
        ("macosx_15_0_arm64", {"a.dylib": (15, 0)}, (True, ["a.dylib"])),
        # A tag that promises an older macOS than the binaries support.
        ("macosx_11_0_arm64", {"a.dylib": (14, 0)}, (False, ["a.dylib"])),
        # A binary that declares no minimum cannot be shown to meet one.
        ("macosx_14_0_arm64", {"a.dylib": None}, (False, ["a.dylib"])),
    ],
)
def test_finds_what_stops_the_wheel_on_the_oldest_supported_macos(
    tag, minimums, violations
):
    assert _macos_minimum_violations(tag, minimums) == violations


# How otool -l prints the load command that holds a library's minimum macOS.
_BUILD_VERSION = """\
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 14.0
      sdk 15.5
   ntools 1
     tool 3
  version 1167.5
"""

# The older command, which ld64 writes instead for targets below macOS 10.14.
_VERSION_MIN = """\
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 10.13
      sdk 15.5
"""


@pytest.mark.parametrize(
    "load_commands,minimum",
    [
        (_BUILD_VERSION, (14, 0)),
        (_VERSION_MIN, (10, 13)),
        # A file with neither command cannot show the minimum is met.
        ("      cmd LC_SEGMENT_64\n", None),
    ],
)
def test_reads_the_minimum_macos_of_a_mach_o_file(load_commands, minimum):
    assert _mach_o_minimum(load_commands) == minimum


@pytest.mark.parametrize(
    "data,minimum",
    [
        # Newer compilers add a version to the architecture. The newest target counts.
        (b"air64_v26-apple-macosx14.0.0\0air64_v26-apple-macosx15.0.0\0", (15, 0)),
        # A Metal library that names no target cannot show the minimum is met.
        (b"MTLB" + b"\0" * 64, None),
    ],
)
def test_reads_the_minimum_macos_of_a_metal_library(data, minimum):
    assert _metallib_minimum(data) == minimum


# Fake wheel contents by path. A Mach-O file is its header followed by the minimum
# macOS of each architecture, which the fake otool prints back.
_THIN, _FAT64 = b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbf"
_WHEEL = {
    "executorch/lib/libexecutorch.dylib": _THIN + b"14.0",
    "executorch/data/bin/flatc": _FAT64 + b"14.0 14.0",
    "executorch/lib/mlx.metallib": b"MTLB air64-apple-macosx14.0.0",
}


@pytest.mark.parametrize(
    "tag,files,error",
    [
        ("macosx_14_0_arm64", _WHEEL, None),
        # The second architecture of a universal executable needs macOS 14.5.
        (
            "macosx_14_0_arm64",
            {**_WHEEL, "executorch/data/bin/flatc": _FAT64 + b"14.0 14.5"},
            r"flatc \(14\.5\)",
        ),
        (
            "macosx_14_0_arm64",
            {**_WHEEL, "executorch/lib/mlx.metallib": b"MTLB air64-apple-macosx14.5.0"},
            r"mlx\.metallib \(14\.5\)",
        ),
        ("macosx_15_0_arm64", _WHEEL, "tagged for a macOS newer than 14.0"),
        (
            "macosx_14_0_arm64",
            {**_WHEEL, "executorch/lib/libexecutorch.dylib": _THIN},
            "otool could not read",
        ),
        (
            "macosx_14_0_arm64",
            {"executorch/lib/libexecutorch.dylib": b"not a Mach-O file"},
            "contains no Mach-O files or Metal libraries",
        ),
    ],
)
def test_checks_every_binary_in_a_macos_wheel(tmp_path, monkeypatch, tag, files, error):
    # The check runs on macOS after a wheel build, but all it reads is a zip file and
    # what lipo and otool print, so here the wheel and both tools are fake.
    wheel = tmp_path / f"executorch-1.0-cp311-cp311-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)

    def run(command, **kwargs):
        if command[0] == "lipo":
            return subprocess.CompletedProcess(command, 0, "arm64\n", "")
        minimums = Path(command[-1]).read_bytes()[4:].decode().split()
        # Without -arch all, otool can print just one architecture of a universal file.
        if "all" not in command:
            minimums = minimums[:1]
        printed = "".join(f"cmd LC_BUILD_VERSION\nminos {m}\n" for m in minimums)
        return subprocess.CompletedProcess(command, 0 if minimums else 1, printed, "")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(test_shared_libraries, "_find_wheel_files", lambda: [wheel])
    monkeypatch.setattr(test_shared_libraries, "_tool", lambda name: name)
    monkeypatch.setattr(subprocess, "run", run)
    if error is None:
        test_shared_libraries.test_wheel_platform_tag()
    else:
        with pytest.raises(AssertionError, match=error):
            test_shared_libraries.test_wheel_platform_tag()
