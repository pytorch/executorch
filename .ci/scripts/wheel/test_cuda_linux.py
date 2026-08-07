#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for a CUDA wheel row.

Runs the same checks a CPU wheel gets, then the ones a GPU wheel needs on top. The extra
checks exist because a GPU wheel can install cleanly, import cleanly, and still be unusable:

  the CUDA libraries can be absent while the wheel is still named as a CUDA build
  the runtime dependency can be undeclared, so a user has nothing to resolve it from
  the loader path can point at the build machine's toolkit, which no user has
  the device code can cover no GPU the row claims, which only appears when a model runs

The build machines for these rows have no GPU, so this does not execute a model. It verifies
everything that can be checked from the artifact, and the release gate runs a model on real
hardware.
"""

import platform
import subprocess
import tempfile
from pathlib import Path

import test_base
import test_cpp_sdk
import test_shared_libraries
from examples.models import Backend, Model


def _package_dir() -> Path:
    import executorch

    return Path(executorch.__path__[0])


def test_cuda_libraries_are_shipped() -> None:
    """The row is named for CUDA, so the CUDA libraries have to be in it."""
    lib_dir = _package_dir() / "lib"
    shipped = {path.name for path in lib_dir.iterdir()} if lib_dir.is_dir() else set()
    expected = {
        "libexecutorch_backend_cuda.so",
        "libexecutorch_extension_cuda.so",
    }
    missing = sorted(expected - shipped)
    assert not missing, (
        f"this is a CUDA row but {missing} are not in the wheel, so it would install as a "
        f"CUDA build with no CUDA delegate. Shipped: {sorted(shipped)}"
    )
    print(f"✓ the CUDA libraries ship ({len(expected)} of them)")


def test_cuda_runtime_is_declared() -> None:
    """The wheel links the CUDA runtime without bundling it, so it must declare it.

    Without this a user installs the wheel and has nothing to resolve libcudart from, which
    surfaces as a loader error at the first import rather than as a resolution failure at
    install time.
    """
    import importlib.metadata as metadata

    requirements = metadata.requires("executorch") or []
    cuda = [
        requirement
        for requirement in requirements
        if "nvidia" in requirement.lower() or "cuda" in requirement.lower()
    ]
    assert cuda, (
        "this is a CUDA row but the wheel declares no CUDA runtime dependency, so nothing "
        "would install the libraries its delegate links"
    )
    print(f"✓ the CUDA runtime is declared ({len(cuda)} requirements)")


def test_cuda_libraries_resolve_relatively() -> None:
    """Each CUDA library must reach its runtime through a relative path.

    An absolute toolkit path names the machine that built the wheel. It resolves there and
    nowhere else, so the wheel would work only on a builder.
    """
    readelf = test_shared_libraries._tool("readelf")
    assert readelf is not None, "readelf is required to inspect the wheel"

    lib_dir = _package_dir() / "lib"
    names = ("libexecutorch_backend_cuda.so", "libexecutorch_extension_cuda.so")
    present = [name for name in names if (lib_dir / name).is_file()]
    # Without this the loop below finds nothing on a wheel that ships no CUDA library and
    # reports a pass, which is the same as having no check at all.
    assert present, (
        f"none of {list(names)} is in the wheel, so this check inspected nothing. A CUDA row "
        "must ship the libraries it is named for."
    )
    for name in present:
        library = lib_dir / name
        output = subprocess.run(
            [readelf, "-d", str(library)], capture_output=True, text=True, check=True
        ).stdout
        needs_cuda = any(
            "NEEDED" in line and "libcud" in line for line in output.splitlines()
        )
        if not needs_cuda:
            print(f"- {name} does not link the CUDA runtime, nothing to resolve")
            continue
        entries: list[str] = []
        for line in output.splitlines():
            if "RPATH" in line or "RUNPATH" in line:
                entries = line.split("[", 1)[1].rstrip("]").strip().split(":")
        relative = [
            entry
            for entry in entries
            if entry.startswith("$ORIGIN") and "nvidia" in entry
        ]
        assert relative, (
            f"{name} links the CUDA runtime but has no relative path to the CUDA wheels "
            f"installed beside it, so it can only resolve where the builder had a toolkit: "
            f"{entries}"
        )
        print(f"✓ {name} resolves the CUDA runtime relatively ({relative[0]})")


if __name__ == "__main__":
    assert platform.system() == "Linux", "the CUDA rows are Linux only"

    test_cuda_libraries_are_shipped()
    test_cuda_runtime_is_declared()
    test_cuda_libraries_resolve_relatively()

    # Everything a CPU wheel is held to still applies: one owner per component, no
    # build-tree paths, and a C++ application able to link what the wheel ships.
    with tempfile.TemporaryDirectory() as work_dir:
        test_shared_libraries.run_tests(Path(work_dir))
    with tempfile.TemporaryDirectory() as work_dir:
        test_cpp_sdk.run_tests(Path(work_dir))

    test_base.run_tests(
        model_tests=[
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
        ]
    )
