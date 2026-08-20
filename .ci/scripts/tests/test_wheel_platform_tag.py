# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the wheel platform tag comparison.

Here rather than in the wheel checks, because the decision under test is a pure
function of two strings. Running it as part of the wheel checks meant it needed eleven
built wheels to exercise one comparison, and it still could not run on a machine that
had not built one.

The comparison had a real defect that this covers. The release pipeline builds in a
manylinux image and rewrites the wheel's file name, so the tag on the file and the tag
auditwheel reports never agree in spelling, and comparing them as text rejected every
correct wheel. No local build reproduces that rewrite, so nothing short of a unit test
catches it before CI.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wheel"))

from test_shared_libraries import (  # noqa: E402
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
