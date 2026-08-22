#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import platform
import ssl
import stat
import sys
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import buck_util
import certifi
import zstd

"""
Locate or download the version of buck2 needed to build ExecuTorch.
It is intended to be invoked from the CMake build logic, and it returns
the path to 'buck2' via stdout. Log messages are written to stderr.

It uses the following logic, in order of precedence, to locate or download
buck2:

 1) If BUCK2 argument is set explicitly, use it. Warn if the version is
    incorrect.
 2) Look for a binary named buck2 on the system path. Take it if it is
    the correct version.
 3) Check for a previously downloaded buck2 binary (from step 4).
 4) Download and cache correct version of buck2.

"""


# Path to the file containing BUCK2 version (build date) for ExecuTorch.
def _buck_version_path() -> Path:
    return buck_util.repo_root_dir() / ".ci/docker/ci_commit_pins/buck2.txt"


@dataclass
class BuckInfo:
    archive_name: str


# Mapping of os family and architecture to buck2 archive name. The target version is the
# hash given by running 'buck2 --version', which is now consistent across platforms. The
# archive name is the archive file name to download, as seen under
# https://github.com/facebook/buck2/releases/.
#
# To update Buck2, download the appropriate version of buck2 for your platform, run
# 'buck2 --version', and update BUCK_TARGET_VERSION. To add a new platform, add the
# corresponding entry to the platform map below, and if adding new os families or
# architectures, update the platform detection logic in resolve_buck2().
BUCK_TARGET_VERSION = "2025-05-06-201beb86106fecdc84e30260b0f1abb5bf576988"

BUCK_PLATFORM_MAP = {
    ("linux", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-unknown-linux-musl.zst",
    ),
    ("linux", "aarch64"): BuckInfo(
        archive_name="buck2-aarch64-unknown-linux-gnu.zst",
    ),
    ("darwin", "aarch64"): BuckInfo(
        archive_name="buck2-aarch64-apple-darwin.zst",
    ),
    ("darwin", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-apple-darwin.zst",
    ),
    ("windows", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-pc-windows-msvc.exe.zst",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locates or downloads the appropriate version of buck2.",
    )
    parser.add_argument(
        "--buck2",
        default="",
        help="Optional user-provided 'buck2' path. If provided, it will be "
        "used. If the version is incorrect, a warning will be logged.",
    )
    parser.add_argument(
        "--cache_dir",
        help="Directory to cache downloaded versions of buck2.",
    )
    return parser.parse_args()


# Returns the path to buck2 on success or a return code on failure.
def resolve_buck2(args: argparse.Namespace) -> Union[str, int]:
    # Find buck2, in order of priority:
    #  1) Explicit buck2 argument.
    #  2) Cached buck2 (previously downloaded).
    #  3) Download buck2.

    # Read the target version (build date) from the CI pin file. Note that
    # this path is resolved relative to the directory containing this script.
    with open(_buck_version_path().absolute().as_posix()) as f:
        target_buck_version = f.read().strip()

    # Determine the target buck2 version string according to the current
    # platform. If the platform isn't linux or darwin, we won't perform
    # any version validation.
    machine = platform.machine().lower()
    arch = "unknown"
    if machine == "x86" or machine == "x86_64" or machine == "amd64":
        arch = "x86_64"
    elif machine == "arm64" or machine == "aarch64":
        arch = "aarch64"

    os_family = "unknown"
    if sys.platform.startswith("linux"):
        os_family = "linux"
    elif sys.platform.startswith("darwin"):
        os_family = "darwin"
    elif sys.platform.startswith("win"):
        os_family = "windows"

    platform_key = (os_family, arch)
    if platform_key not in BUCK_PLATFORM_MAP:
        print(
            f"Unknown platform {platform_key}. Buck2 binary must be downloaded manually.",
            file=sys.stderr,
        )
        return args.buck2 or "buck2"

    buck_info = BUCK_PLATFORM_MAP[platform_key]

    if args.buck2:
        # If we have an explicit buck2 arg, check the version and fail if
        # there is a mismatch.
        ver = buck_util.get_buck2_version(args.buck2)
        if ver == BUCK_TARGET_VERSION:
            return args.buck2
        else:
            print(
                f'The provided buck2 binary "{args.buck2}" reports version '
                f'"{ver}", but ExecuTorch needs version '
                f'"{BUCK_TARGET_VERSION}". Ensure that the correct buck2'
                " version is installed or avoid explicitly passing the BUCK2 "
                "version to automatically download the correct version.",
                file=sys.stderr,
            )

            # Return an error, since the build will fail later. This lets us
            # give the user a more useful error message. Note that an exit
            # code of 2 allows us to distinguish from an unexpected error,
            # such as a failed import, which exits with 1.
            return 2
    else:
        # Download buck2 or used previously cached download.
        cache_dir = Path(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        buck2_local_path = (
            (cache_dir / f"buck2-{BUCK_TARGET_VERSION}").absolute().as_posix()
        )

        # Check for a previously cached buck2 binary. The filename includes
        # the version hash, so we don't have to worry about using an
        # outdated binary, in the event that the target version is updated.
        if os.path.isfile(buck2_local_path):
            return buck2_local_path

        buck2_archive_url = f"https://github.com/facebook/buck2/releases/download/{target_buck_version}/{buck_info.archive_name}"

        print(f"Downloading buck2 from {buck2_archive_url}...", file=sys.stderr)
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(buck2_archive_url, context=ssl_context) as request:
            # Extract and chmod.
            data = request.read()
            decompressed_bytes = zstd.decompress(data)

            with open(buck2_local_path, "wb") as f:
                f.write(decompressed_bytes)

        file_stat = os.stat(buck2_local_path)
        os.chmod(buck2_local_path, file_stat.st_mode | stat.S_IEXEC)

    return buck2_local_path


def main():
    args = parse_args()
    resolved_path_or_error = resolve_buck2(args)
    if isinstance(resolved_path_or_error, str):
        print(resolved_path_or_error)
    else:
        sys.exit(resolved_path_or_error)


if __name__ == "__main__":
    main()
