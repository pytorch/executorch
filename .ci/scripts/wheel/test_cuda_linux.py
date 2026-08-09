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

import os
import pathlib
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

    Every shipped library that links the CUDA runtime is inspected, wherever it lives. Naming
    only the two in lib/ skipped libaoti_cuda_shims.so, which sits under backends/cuda/, links
    cudart and curand, and carries the device code, so an absolute toolkit path on the library
    that matters most shipped green.
    """
    readelf = test_shared_libraries._tool("readelf")
    assert readelf is not None, "readelf is required to inspect the wheel"

    package_dir = _package_dir()
    libraries = sorted(test_shared_libraries._shipped_shared_objects(package_dir))
    # Without this the loop below finds nothing on a wheel that ships no CUDA library and
    # reports a pass, which is the same as having no check at all.
    assert libraries, f"no shared libraries found under {package_dir}"

    linked_to_cuda = []
    for library in libraries:
        output = subprocess.run(
            [readelf, "-d", str(library)], capture_output=True, text=True, check=True
        ).stdout
        if any("NEEDED" in line and "libcud" in line for line in output.splitlines()):
            linked_to_cuda.append((library, output))

    assert linked_to_cuda, (
        "no shipped library links the CUDA runtime, so this check inspected nothing. A CUDA "
        "row must ship the libraries it is named for."
    )
    for library, output in linked_to_cuda:
        name = library.relative_to(package_dir)
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


def _row_architectures() -> list[str]:
    """The architectures this row claims, from the same script the build uses.

    A refusal from that script is a fault, not an absence. It returns non-zero when a CUDA row reaches it
    with no version, which is precisely the case that would otherwise build device code for whatever GPU the
    builder happens to have, so swallowing it here would hide the one failure this check exists to catch.

    EXECUTORCH_BUILD_CUDA is passed through because that is how the build invokes the script, and the
    refusal is conditional on it. Without it the script returned an empty list on a CUDA row that had lost
    its version, this check reported nothing to do, and the assertion below could never fire.
    """
    script = pathlib.Path(__file__).parent / "cuda_arch_list.sh"
    assert script.is_file(), f"the architecture script is missing at {script}"
    result = subprocess.run(
        ["bash", "-c", f"source {script}; executorch_cuda_arch_list"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "EXECUTORCH_BUILD_CUDA": "1"},
    )
    assert result.returncode == 0, (
        f"the architecture script refused this row with exit {result.returncode}, so the build had no list "
        f"to compile against: {result.stderr.strip()[:300]}"
    )
    # "8.0 9.0" describes sm_80 and sm_90.
    return ["sm_" + value.replace(".", "") for value in result.stdout.split()]


def test_device_code_covers_the_row() -> None:
    """Every GPU the row claims must have device code in the shipped libraries.

    A row that promises a GPU it did not compile for produces a wheel that installs and then dies
    at the first kernel launch, which is the worst failure to publish.
    """
    expected = _row_architectures()
    if not expected:
        print("- this row claims no GPU architectures, nothing to check")
        return

    cuobjdump = test_shared_libraries._tool("cuobjdump")
    if cuobjdump is None:
        raise AssertionError(
            "cuobjdump is required to check device code, and this is a CUDA row. Without it a "
            "wheel missing code for a claimed GPU would ship unnoticed."
        )

    # Searched across every shipped library rather than a named one. The kernels are compiled
    # into their own library, not into the delegate, and which library holds them is an internal
    # detail. What the row promises is that the wheel covers those GPUs.
    present: set[str] = set()
    inspected = []
    for library in sorted(_package_dir().rglob("*.so")):
        listed = subprocess.run(
            [cuobjdump, "--list-elf", str(library)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        found = {
            token
            for token in listed.replace(".", " ").split()
            if token.startswith("sm_")
        }
        if found:
            inspected.append(f"{library.name} ({', '.join(sorted(found))})")
            present |= found

    assert inspected, (
        "no shipped library contains any GPU device code, so this wheel cannot run a model on any "
        f"GPU, while the row claims {expected}"
    )
    missing = sorted(set(expected) - present)
    assert not missing, (
        f"the row claims {expected} but the wheel carries no device code for {missing}. "
        f"Found: {inspected}. A user with one of those GPUs would install this wheel and fail at "
        "the first kernel launch."
    )
    print(f"✓ device code covers the row: {inspected}")


def test_the_delegate_registers() -> None:
    """The delegate has to appear in the runtime's backend list, not merely be present as a file.

    Registration happens in a static initializer, which a normal link discards because nothing in the
    program references it. Keeping it alive needs a linker option, and a wheel whose delegate ships but
    does not register would load a delegated program and fail with an unregistered backend. That is the
    failure this whole layout is most able to introduce, so it is worth asserting rather than assuming.

    Needs no GPU: registration is a link-time property, checked by importing.
    """
    from executorch.extension.pybindings.portable_lib import (
        _get_registered_backend_names,
    )

    registered = _get_registered_backend_names()
    assert "CudaBackend" in registered, (
        f"the wheel ships the CUDA delegate but CudaBackend is not registered: {registered}. "
        "The library is present and its static initializer did not run, which means the option "
        "that keeps it on the link line stopped working."
    )
    print(f"✓ the delegate registers: CudaBackend among {len(registered)} backend(s)")


if __name__ == "__main__":
    assert platform.system() == "Linux", "the CUDA rows are Linux only"

    test_cuda_libraries_are_shipped()
    test_cuda_runtime_is_declared()
    test_cuda_libraries_resolve_relatively()
    test_device_code_covers_the_row()
    test_the_delegate_registers()

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
