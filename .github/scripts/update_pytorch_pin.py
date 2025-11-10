#!/usr/bin/env python3

import base64
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path


def parse_nightly_version(nightly_version):
    """
    Parse NIGHTLY_VERSION (e.g., 'dev20251004') to date string (e.g., '2025-10-04').

    Args:
        nightly_version: String in format 'devYYYYMMDD'

    Returns:
        Date string in format 'YYYY-MM-DD'
    """
    match = re.match(r"dev(\d{4})(\d{2})(\d{2})", nightly_version)
    if not match:
        raise ValueError(f"Invalid NIGHTLY_VERSION format: {nightly_version}")

    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def get_torch_nightly_version():
    """
    Read NIGHTLY_VERSION from torch_pin.py.

    Returns:
        NIGHTLY_VERSION string
    """
    with open("torch_pin.py", "r") as f:
        content = f.read()

    match = re.search(r'NIGHTLY_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find NIGHTLY_VERSION in torch_pin.py")

    return match.group(1)


def get_commit_hash_for_nightly(date_str):
    """
    Fetch commit hash from PyTorch nightly branch for a given date.

    Args:
        date_str: Date string in format 'YYYY-MM-DD'

    Returns:
        Commit hash string
    """
    api_url = "https://api.github.com/repos/pytorch/pytorch/commits"
    params = f"?sha=nightly&per_page=50"
    url = api_url + params

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "ExecuTorch-Bot")

    try:
        with urllib.request.urlopen(req) as response:
            commits = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching commits: {e}", file=sys.stderr)
        sys.exit(1)

    # Look for commit with title matching "{date_str} nightly release"
    target_title = f"{date_str} nightly release"

    for commit in commits:
        commit_msg = commit.get("commit", {}).get("message", "")
        # Check if the first line of commit message matches
        first_line = commit_msg.split("\n")[0].strip()
        if first_line.startswith(f"{date_str} nightly"):
            return extract_hash_from_title(first_line)

    raise ValueError(
        f"Could not find commit with title matching '{target_title}' in nightly branch"
    )


def extract_hash_from_title(title):
    match = re.search(r"\(([0-9a-fA-F]{7,40})\)", title)
    if not match:
        raise ValueError(f"Could not extract commit hash from title '{title}'")
    return match.group(1)


def update_pytorch_pin(commit_hash):
    """
    Update .ci/docker/ci_commit_pins/pytorch.txt with the new commit hash.

    Args:
        commit_hash: Commit hash to write
    """
    pin_file = ".ci/docker/ci_commit_pins/pytorch.txt"
    with open(pin_file, "w") as f:
        f.write(f"{commit_hash}\n")
    print(f"Updated {pin_file} with commit hash: {commit_hash}")


def should_skip_file(filename):
    """
    Check if a file should be skipped during sync (build files).

    Args:
        filename: Base filename to check

    Returns:
        True if file should be skipped
    """
    skip_files = {"BUCK", "CMakeLists.txt", "TARGETS", "targets.bzl"}
    return filename in skip_files


def fetch_file_content(commit_hash, file_path):
    """
    Fetch file content from GitHub API.

    Args:
        commit_hash: Commit hash to fetch from
        file_path: File path in the repository

    Returns:
        File content as bytes
    """
    api_url = f"https://api.github.com/repos/pytorch/pytorch/contents/{file_path}?ref={commit_hash}"

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "ExecuTorch-Bot")

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            # Content is base64 encoded
            content = base64.b64decode(data["content"])
            return content
    except urllib.request.HTTPError as e:
        print(f"Error fetching file {file_path}: {e}", file=sys.stderr)
        raise


def sync_directory(et_dir, pt_path, commit_hash):
    """
    Sync files from PyTorch to ExecuTorch using GitHub API.
    Only syncs files that already exist in ExecuTorch - does not add new files.

    Args:
        et_dir: ExecuTorch directory path
        pt_path: PyTorch directory path in the repository (e.g., "c10")
        commit_hash: Commit hash to fetch from

    Returns:
        Number of files grafted
    """
    files_grafted = 0
    print(f"Checking {et_dir} vs pytorch/{pt_path}...")

    if not et_dir.exists():
        print(f"Warning: ExecuTorch directory {et_dir} does not exist, skipping")
        return 0

    # Loop through files in ExecuTorch directory
    for et_file in et_dir.rglob("*"):
        if not et_file.is_file():
            continue

        # Skip build files
        if should_skip_file(et_file.name):
            continue

        # Construct corresponding path in PyTorch
        rel_path = et_file.relative_to(et_dir)
        pt_file_path = f"{pt_path}/{rel_path}".replace("\\", "/")

        # Fetch content from PyTorch and compare
        try:
            pt_content = fetch_file_content(commit_hash, pt_file_path)
            et_content = et_file.read_bytes()

            if pt_content != et_content:
                print(f"⚠️  Difference detected in {rel_path}")
                print(f"📋 Grafting from PyTorch commit {commit_hash}...")

                et_file.write_bytes(pt_content)
                print(f"✅ Grafted {et_file}")
                files_grafted += 1
        except urllib.request.HTTPError as e:
            if e.code != 404:  # It's ok to have more files in ET than pytorch/pytorch.
                print(f"Error fetching {rel_path} from PyTorch: {e}")
        except Exception as e:
            print(f"Error syncing {rel_path}: {e}")
            continue

    return files_grafted


def sync_c10_directories(commit_hash):
    """
    Sync c10 and torch/headeronly directories from PyTorch to ExecuTorch using GitHub API.

    Args:
        commit_hash: PyTorch commit hash to sync from

    Returns:
        Total number of files grafted
    """
    print("\n🔄 Syncing c10 directories from PyTorch via GitHub API...")

    # Get repository root
    repo_root = Path.cwd()

    # Define directory pairs to sync (from check_c10_sync.sh)
    # Format: (executorch_dir, pytorch_path_in_repo)
    dir_pairs = [
        (
            repo_root / "runtime/core/portable_type/c10/c10",
            "c10",
        ),
        (
            repo_root / "runtime/core/portable_type/c10/torch/headeronly",
            "torch/headeronly",
        ),
    ]

    total_grafted = 0
    for et_dir, pt_path in dir_pairs:
        files_grafted = sync_directory(et_dir, pt_path, commit_hash)
        total_grafted += files_grafted

    if total_grafted > 0:
        print(f"\n✅ Successfully grafted {total_grafted} file(s) from PyTorch")
    else:
        print("\n✅ No differences found - c10 is in sync")

    return total_grafted


def main():
    try:
        # Read NIGHTLY_VERSION from torch_pin.py
        nightly_version = get_torch_nightly_version()
        print(f"Found NIGHTLY_VERSION: {nightly_version}")

        # Parse to date string
        date_str = parse_nightly_version(nightly_version)
        print(f"Parsed date: {date_str}")

        # Fetch commit hash from PyTorch nightly branch
        commit_hash = get_commit_hash_for_nightly(date_str)
        print(f"Found commit hash: {commit_hash}")

        # Update the pin file
        update_pytorch_pin(commit_hash)

        # Sync c10 directories from PyTorch
        sync_c10_directories(commit_hash)

        print(
            "\n✅ Successfully updated PyTorch commit pin and synced c10 directories!"
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
