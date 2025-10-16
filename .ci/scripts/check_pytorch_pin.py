#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script checks if the PyTorch commit hash in pytorch.txt matches
the commit hash for the NIGHTLY_VERSION specified in torch_pin.py.

It uses functions from update_pytorch_pin.py to fetch the expected commit
hash and compares it with the current pin.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path to import update_pytorch_pin module
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / ".github" / "scripts"))

from update_pytorch_pin import (
    get_commit_hash_for_nightly,
    get_torch_nightly_version,
    parse_nightly_version,
)


def get_current_pytorch_commit():
    """
    Read the current commit hash from .ci/docker/ci_commit_pins/pytorch.txt.

    Returns:
        Current commit hash string
    """
    pin_file = repo_root / ".ci" / "docker" / "ci_commit_pins" / "pytorch.txt"
    if not pin_file.exists():
        raise FileNotFoundError(f"Could not find {pin_file}")

    with open(pin_file, "r") as f:
        commit_hash = f.read().strip()

    if not commit_hash:
        raise ValueError(f"{pin_file} is empty")

    return commit_hash


def main():
    print("=" * 80)
    print("Checking PyTorch commit pin consistency")
    print("=" * 80)
    print()

    try:
        # Get NIGHTLY_VERSION from torch_pin.py
        os.chdir(repo_root)
        nightly_version = get_torch_nightly_version()
        print(f"Nightly version: {nightly_version}")

        # Parse to date string
        date_str = parse_nightly_version(nightly_version)
        print(f"Expected date: {date_str}")

        # Get expected commit hash from PyTorch nightly branch
        print(f"Fetching commit hash for {date_str} from PyTorch nightly branch...")
        expected_commit = get_commit_hash_for_nightly(date_str)
        print(f"Expected commit hash: {expected_commit}")
        print()

        # Get current commit hash from pytorch.txt
        current_commit = get_current_pytorch_commit()
        print(f"Current commit hash: {current_commit}")
        print()

        # Compare commits
        print("=" * 80)
        print("Verification Result")
        print("=" * 80)
        print()

        if expected_commit == current_commit:
            print("✓ SUCCESS: PyTorch commit pin matches the nightly version!")
            print()
            print(f"Commit {current_commit} corresponds to {nightly_version}")
            print()
            print(
                f"Reference: https://hud.pytorch.org/pytorch/pytorch/commit/{current_commit}"
            )
            return 0
        else:
            print("✗ ERROR: PyTorch commit pin does NOT match the nightly version!")
            print()
            print(f"  Expected commit: {expected_commit}")
            print(f"  Current commit:  {current_commit}")
            print()
            print(f"The commit in .ci/docker/ci_commit_pins/pytorch.txt")
            print(f"does not correspond to NIGHTLY_VERSION={nightly_version}")
            print()
            print("To fix this, you can run:")
            print(f"  python .github/scripts/update_pytorch_pin.py")
            print(
                "or manually update the commit hash in .ci/docker/ci_commit_pins/pytorch.txt"
            )
            print(f"with the expected commit hash {expected_commit}")
            return 1

    except Exception as e:
        print(f"✗ ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
