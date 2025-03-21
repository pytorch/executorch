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
import stat
import sys
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import buck_util
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
    target_versions: Sequence[str]


# Mapping of os family and architecture to buck2 binary versions. The
# target version is the hash given by running 'buck2 --version'. The
# archive name is the archive file name to download, as seen under
# https://github.com/facebook/buck2/releases/.
#
# To add or update versions, download the appropriate version of buck2
# and run 'buck2 --version'. Add the corresponding entry to the platform
# map below, and if adding new os families or architectures, update the
# platform detection logic in resolve_buck2().
#
# Some platforms (linux) provide multiple binaries (GNU and MUSL). All
# versions in the list are accepted when validating a user-provided or
# system buck2.
BUCK_PLATFORM_MAP = {
    ("linux", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-unknown-linux-musl.zst",
        target_versions=[
            # MUSL
            "edae27cfca00053d9c5f7c7be81b6b0d7d07573a50be374ce53a9d8692afa5fc",
            # GNU
            "10334cb20cb7c321",
        ],
    ),
    ("linux", "aarch64"): BuckInfo(
        archive_name="buck2-aarch64-unknown-linux-gnu.zst",
        target_versions=[
            # MUSL
            "5d7af382acbe0dde70f0e9b0a0bc36deea906077ec1ffe80d3fa280490109051",
            # GNU
            "08d4382de22fab275978abc7c27c001d7823eb2f",
        ],
    ),
    ("darwin", "aarch64"): BuckInfo(
        archive_name="buck2-aarch64-apple-darwin.zst",
        target_versions=["f3b7a37732803ed090cd8a37f00cc000"],
    ),
    ("darwin", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-apple-darwin.zst",
        target_versions=["9c9a583658d43e82b41f3fc9d369a9b0"],
    ),
    ("windows", "x86_64"): BuckInfo(
        archive_name="buck2-x86_64-pc-windows-msvc.exe.zst",
        target_versions=[
            "c7d378f3f307e9590f0b29a5f7f1b21b8e784f4e4bd30a0160b2a69df50d2ee0"
        ],
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
    #  2) System buck2 (if correct version).
    #  3) Cached buck2 (previously downloaded).
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
        if ver in buck_info.target_versions:
            return args.buck2
        else:
            print(
                f'The provided buck2 binary "{args.buck2}" reports version '
                f'"{ver}", but ExecuTorch needs version '
                f'"{buck_info.target_versions[0]}". Ensure that the correct buck2'
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
        # Look for system buck2 and check version. Note that this can return
        # None.
        ver = buck_util.get_buck2_version("buck2")
        if ver in buck_info.target_versions:
            # Use system buck2.
            return "buck2"
        else:
            # Download buck2 or used previously cached download.
            cache_dir = Path(args.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

            buck2_local_path = (
                (cache_dir / f"buck2-{buck_info.target_versions[0]}")
                .absolute()
                .as_posix()
            )

            # Check for a previously cached buck2 binary. The filename includes
            # the version hash, so we don't have to worry about using an
            # outdated binary, in the event that the target version is updated.
            if os.path.isfile(buck2_local_path):
                return buck2_local_path

            buck2_archive_url = f"https://github.com/facebook/buck2/releases/download/{target_buck_version}/{buck_info.archive_name}"

            try:
                print(f"Downloading buck2 from {buck2_archive_url}...", file=sys.stderr)
                archive_file, _ = urllib.request.urlretrieve(buck2_archive_url)

                # Extract and chmod.
                with open(archive_file, "rb") as f:
                    data = f.read()
                    decompressed_bytes = zstd.decompress(data)

                with open(buck2_local_path, "wb") as f:
                    f.write(decompressed_bytes)

                file_stat = os.stat(buck2_local_path)
                os.chmod(buck2_local_path, file_stat.st_mode | stat.S_IEXEC)
            finally:
                os.remove(archive_file)

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
