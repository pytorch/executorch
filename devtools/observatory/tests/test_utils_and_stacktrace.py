# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from executorch.devtools.observatory.lenses.stack_trace import StackTraceLens
from executorch.devtools.observatory import utils


@pytest.mark.parametrize(
    "remote_url, expected",
    [
        ("git@github.com:pytorch/executorch.git", "https://github.com/pytorch/executorch"),
        ("git@github.qualcomm.com:MLG/executorch", "https://github.qualcomm.com/MLG/executorch"),
        (
            "ssh://git@github.enterprise.local:8022/org/repo.git",
            "https://github.enterprise.local:8022/org/repo",
        ),
        ("https://github.com/pytorch/executorch.git", "https://github.com/pytorch/executorch"),
        ("file:///tmp/executorch", None),
    ],
)
def test_normalize_remote_url(remote_url, expected):
    assert utils._normalize_remote_url(remote_url) == expected  # type: ignore[attr-defined]


def test_stack_trace_uses_commit_links(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_file = repo_root / "pkg" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("line = 1\n")

    frame = SimpleNamespace(
        filename=str(source_file),
        function="test_fn",
        lineno=1,
        code_context=["line = 1\n"],
    )

    git_info = SimpleNamespace(
        commit_blob_url="https://github.com/org/repo/blob/abc123",
        branch_blob_url="https://github.com/org/repo/blob/main",
        github_link="https://github.com/org/repo/tree/main",
    )

    from executorch.devtools.observatory.lenses import stack_trace as stack_trace_mod

    monkeypatch.setattr(stack_trace_mod, "inspect", SimpleNamespace(stack=lambda: [frame]))
    monkeypatch.setattr(stack_trace_mod, "get_repo_root", lambda: str(repo_root))
    monkeypatch.setattr(stack_trace_mod, "get_git_info", lambda: git_info)
    monkeypatch.setattr(stack_trace_mod, "is_in_repo", lambda path: str(path).startswith(str(repo_root)))

    frames = StackTraceLens._get_stack_trace()
    assert len(frames) == 1
    assert frames[0]["link"] == "https://github.com/org/repo/blob/abc123/pkg/module.py#L1"


def test_stack_trace_falls_back_to_branch_links(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_file = repo_root / "pkg" / "fallback.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("line = 2\n")

    frame = SimpleNamespace(
        filename=str(source_file),
        function="fallback_fn",
        lineno=2,
        code_context=["line = 2\n"],
    )

    git_info = SimpleNamespace(
        commit_blob_url=None,
        branch_blob_url="https://github.com/org/repo/blob/main",
        github_link="https://github.com/org/repo/tree/main",
    )

    from executorch.devtools.observatory.lenses import stack_trace as stack_trace_mod

    monkeypatch.setattr(stack_trace_mod, "inspect", SimpleNamespace(stack=lambda: [frame]))
    monkeypatch.setattr(stack_trace_mod, "get_repo_root", lambda: str(repo_root))
    monkeypatch.setattr(stack_trace_mod, "get_git_info", lambda: git_info)
    monkeypatch.setattr(stack_trace_mod, "is_in_repo", lambda path: str(path).startswith(str(repo_root)))

    frames = StackTraceLens._get_stack_trace()
    assert len(frames) == 1
    assert frames[0]["link"] == "https://github.com/org/repo/blob/main/pkg/fallback.py#L2"
