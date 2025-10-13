#!/usr/bin/env python3

import json
import re
import sys
import urllib.request
from datetime import datetime


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
    params = f"?sha=nightly&per_page=100"
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
        if first_line == target_title or first_line.startswith(f"{date_str} nightly"):
            return commit["sha"]

    raise ValueError(
        f"Could not find commit with title matching '{target_title}' in nightly branch"
    )


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

        print("Successfully updated PyTorch commit pin!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
