# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class GitInfo:
    """Git repository information for source links."""

    remote_url: Optional[str] = None
    remote_https_url: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    is_dirty: bool = False
    github_link: Optional[str] = None
    branch_blob_url: Optional[str] = None
    commit_blob_url: Optional[str] = None


_cached_git_info: Optional[GitInfo] = None
_cached_repo_root: Optional[str] = None


def _strip_git_suffix(value: str) -> str:
    return value[:-4] if value.endswith(".git") else value


def _normalize_remote_url(remote_url: str) -> Optional[str]:
    """Return an https URL for browser links regardless of git remote scheme."""

    if not remote_url:
        return None

    remote_url = remote_url.strip()
    if not remote_url:
        return None

    if remote_url.startswith("git@"):
        try:
            user_host, path = remote_url.split(":", 1)
        except ValueError:
            return None
        host = user_host.split("@", 1)[1]
        cleaned_path = _strip_git_suffix(path.lstrip("/").rstrip("/"))
        return f"https://{host}/{cleaned_path}" if cleaned_path else f"https://{host}"

    if remote_url.startswith("ssh://"):
        parsed = urlparse(remote_url)
        host = parsed.hostname
        if not host:
            return None
        port_suffix = f":{parsed.port}" if parsed.port and parsed.port not in (22,) else ""
        cleaned_path = _strip_git_suffix(parsed.path.lstrip("/").rstrip("/"))
        return f"https://{host}{port_suffix}/{cleaned_path}" if cleaned_path else f"https://{host}{port_suffix}"

    parsed = urlparse(remote_url)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname
        if not host:
            return None
        port_suffix = ""
        if parsed.port and not (parsed.scheme == "http" and parsed.port == 80) and not (parsed.scheme == "https" and parsed.port == 443):
            port_suffix = f":{parsed.port}"
        cleaned_path = _strip_git_suffix(parsed.path.lstrip("/").rstrip("/"))
        return f"https://{host}{port_suffix}/{cleaned_path}" if cleaned_path else f"https://{host}{port_suffix}"

    return None


def get_repo_root() -> Optional[str]:
    """Return repository root, or None when unavailable."""

    global _cached_repo_root
    if _cached_repo_root is not None:
        return _cached_repo_root

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            _cached_repo_root = result.stdout.strip()
            return _cached_repo_root
    except Exception as exc:
        logging.debug("[Observatory] Failed to detect repo root: %s", exc)

    return None


def get_git_info() -> GitInfo:
    """Return git metadata used for stack trace source links."""

    global _cached_git_info
    if _cached_git_info is not None:
        return _cached_git_info

    info = GitInfo()

    try:
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            capture_output=True,
            text=True,
            check=False,
        )

        remote_name = None
        if upstream.returncode == 0:
            upstream_name = upstream.stdout.strip()
            if "/" in upstream_name:
                remote_name, info.branch = upstream_name.split("/", 1)

        if info.branch is None:
            local = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if local.returncode == 0:
                info.branch = local.stdout.strip()
                remote_name = "origin"

        if remote_name:
            remote_url = subprocess.run(
                ["git", "config", "--get", f"remote.{remote_name}.url"],
                capture_output=True,
                text=True,
                check=False,
            )
            if remote_url.returncode == 0:
                info.remote_url = remote_url.stdout.strip()
                info.remote_https_url = _normalize_remote_url(info.remote_url)

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            info.commit_hash = commit.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.returncode == 0:
            info.is_dirty = bool(dirty.stdout.strip())

        if info.remote_https_url and info.branch:
            info.github_link = f"{info.remote_https_url}/tree/{info.branch}"
            info.branch_blob_url = f"{info.remote_https_url}/blob/{info.branch}"
        if info.remote_https_url and info.commit_hash:
            info.commit_blob_url = f"{info.remote_https_url}/blob/{info.commit_hash}"
    except Exception as exc:
        logging.debug("[Observatory] Failed to query git info: %s", exc)

    _cached_git_info = info
    return info


def is_in_repo(filepath: str) -> bool:
    """Check whether filepath is inside the current repository root."""

    repo_root = get_repo_root()
    if repo_root is None:
        return False

    try:
        abs_path = os.path.abspath(filepath)
        abs_root = os.path.abspath(repo_root)
        return abs_path.startswith(abs_root)
    except Exception:
        return False
