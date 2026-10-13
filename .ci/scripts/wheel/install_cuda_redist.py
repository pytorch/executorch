#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assemble part of a CUDA toolkit from NVIDIA's redistributable component archives.

The Windows CUDA wheel end-to-end jobs cover several CUDA trains while the CI images carry
one toolkit, and NVIDIA publishes no full Windows installer for every release. The redist
archives exist for every release on both platforms, need no installer or admin rights, and
are checksummed in a JSON index. Merging the components a job needs into one directory
gives the layout nvcc, CMake's FindCUDAToolkit and torch's cross-compile step expect.

    install_cuda_redist.py --train 13.4 --platform linux-x86_64 --dest /opt/cuda-13.4
    install_cuda_redist.py --train 13.4 --platform windows-x86_64 --dest DIR --components cuda_cudart
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"

# The patch release each train resolves to: one entry per train in filter_cuda_matrix.py's
# SUPPORTED_CUDA_VERSIONS, which the end-to-end jobs take their matrix from, and 13.0, which
# is no longer published but still builds from source and is what local tests of it use. A
# train missing here fails the job at argument parsing rather than testing a different
# release.
_RELEASES = {"13.0": "13.0.2", "13.2": "13.2.2", "13.4": "13.4.1"}

# What a build of the CUDA backend needs. "a|b" names one component a release publishes
# under either name (CCCL was renamed between releases).
_BUILD_COMPONENTS = [
    "cuda_nvcc",
    "cuda_cudart",
    "cuda_crt",
    "cccl|cuda_cccl",
    "libnvvm",
    "cuda_cuobjdump",
    "cuda_nvrtc",
    "cuda_profiler_api",
    "cuda_nvtx",
    # rand.cu includes curand_kernel.h.
    "libcurand",
]


def _fetch(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response, open(
        destination, "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)


def _extract(archive: Path, into: Path) -> Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
    else:
        with tarfile.open(archive) as bundle:
            if hasattr(tarfile, "fully_trusted_filter"):
                bundle.extractall(into, filter="fully_trusted")
            else:
                bundle.extractall(into)
    # Each archive holds one top-level directory named after itself.
    (top,) = [entry for entry in into.iterdir() if entry.is_dir()]
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", required=True, choices=sorted(_RELEASES))
    parser.add_argument(
        "--platform", required=True, choices=["linux-x86_64", "windows-x86_64"]
    )
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--components", nargs="+", default=_BUILD_COMPONENTS)
    arguments = parser.parse_args()

    release = _RELEASES[arguments.train]
    with urllib.request.urlopen(
        f"{_REDIST}/redistrib_{release}.json", timeout=120
    ) as response:
        index = json.load(response)
    arguments.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for choice in arguments.components:
            names = [
                name
                for name in choice.split("|")
                if arguments.platform in index.get(name, {})
            ]
            if not names:
                sys.exit(
                    f"CUDA {release} publishes none of {choice} for {arguments.platform}"
                )
            entry = index[names[0]][arguments.platform]
            archive = work / Path(entry["relative_path"]).name
            _fetch(f"{_REDIST}/{entry['relative_path']}", archive)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                sys.exit(f"checksum mismatch for {archive.name}")
            staging = work / "staging"
            top = _extract(archive, staging)
            shutil.copytree(top, arguments.dest, symlinks=True, dirs_exist_ok=True)
            shutil.rmtree(staging)
            archive.unlink()
            print(
                f"merged {names[0]} {index[names[0]]['version']} into {arguments.dest}"
            )
    # The Linux archives keep libraries in lib/, while nvcc's link step and CMake's
    # FindCUDAToolkit look in lib64/, which is where an installed toolkit has them.
    lib, lib64 = arguments.dest / "lib", arguments.dest / "lib64"
    if arguments.platform == "linux-x86_64" and lib.is_dir() and not lib64.exists():
        lib64.symlink_to("lib", target_is_directory=True)


if __name__ == "__main__":
    main()
