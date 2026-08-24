#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for a CUDA wheel row.

Runs the checks a GPU wheel needs, then the packaging, backend, and C++ SDK checks a CPU wheel
gets. The extra CUDA checks exist because a GPU wheel can install cleanly, import cleanly, and
still be unusable:

  the CUDA libraries can be absent while the wheel is still named as a CUDA build
  the runtime dependency can be undeclared, so a user has nothing to resolve it from
  the loader path can point at the build machine's toolkit, which no user has
  the device code can cover no GPU the row claims, which only appears when a model runs

A model-execution check runs where a device exists. The x86_64 rows land on a GPU runner, so
a model is exported through the CUDA partitioner and run there, and its output is compared
against eager. The aarch64 rows have no accelerator on their validation runner, so that check
skips and prints why, which keeps a green result on those rows from standing for work that did
not happen. Everything else here is inspection of the shipped artifacts, which needs no device.
"""

import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Set

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
    cudart, and carries the device code, so an absolute toolkit path on the library
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
                entries += line.split("[", 1)[1].rstrip("]").strip().split(":")
        # Resolved against the library's own directory rather than pattern-matched. Accepting any
        # entry that merely starts with $ORIGIN and mentions nvidia pins the presence of a hop, not
        # that the hop lands anywhere: $ORIGIN/nvidia-does-not-exist satisfied the old form. A hop
        # at the wrong depth is exactly the defect this check exists to catch.
        relative = [
            entry
            for entry in entries
            if entry.startswith("$ORIGIN") and "nvidia" in entry
            # Stat last, so the filesystem is only touched for an entry that could match.
            and (library.parent / entry.replace("$ORIGIN/", "").replace("$ORIGIN", "."))
            .resolve()
            .is_dir()
        ]
        assert relative, (
            f"{name} links the CUDA runtime but records no relative path that RESOLVES to an "
            f"installed CUDA wheel directory, so it can only resolve where the builder had a "
            f"toolkit: "
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
    with_device_code: dict = {}
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
            with_device_code[library.name] = found

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
    # The other direction matters just as much. Device code for an architecture the row does not
    # claim means the build did not use the row's list, so whatever selected the architectures
    # ignored it. That went unnoticed once already: a selection bug substituted a single default
    # architecture and this check stayed green because it only looked for what was absent.
    unexpected = sorted(present - set(expected))
    assert not unexpected, (
        f"the row claims {sorted(set(expected))} but the wheel also carries device code for "
        f"{unexpected}. Found: {inspected}. The build did not use the row's list, so the artifact "
        "does not match what the row published."
    )
    # Every library that carries device code has to cover the row on its own. Unioning
    # across libraries let a library with kernels cover only part of the row while an
    # unrelated object supplied the rest, so on a GPU the first one did not compile for
    # there was no executable kernel even though the union looked complete.
    short = sorted(set(expected))
    for library in sorted(with_device_code):
        library_missing = sorted(set(expected) - with_device_code[library])
        assert not library_missing, (
            f"{library} carries GPU device code but none for {library_missing}, while the row "
            f"claims {short}. Checking the union across libraries hid this: another shipped "
            "object supplied those architectures, and on such a GPU this library would have no "
            "executable kernel."
        )
    print(f"✓ device code covers the row in every library that has any: {inspected}")


def test_portable_device_code_is_present() -> None:
    """The newest architecture must also ship in its portable form.

    The build appends "+PTX" for the top architecture so a GPU newer than any in the row can
    still run, by having the driver compile that portable form at load time. Without it such a
    GPU gets no usable code at all.

    Checked with --list-ptx rather than --list-elf. --list-elf prints byte-identical output for
    a library built with or without the portable form, so it cannot see this. --list-ptx prints
    an entry only for the library that has it. The entry is named for the target architecture,
    "sm_90.ptx" rather than "compute_90.ptx", which is what the real tool prints.
    """
    expected = _row_architectures()
    if not expected:
        print("- this row claims no GPU architectures, nothing to check")
        return

    cuobjdump = test_shared_libraries._tool("cuobjdump")
    assert (
        cuobjdump is not None
    ), "cuobjdump is required to check the portable device code, and this is a CUDA row."

    # The newest architecture in the row, which is the one the build makes portable.
    newest = max(expected, key=lambda name: int(name.removeprefix("sm_")))

    found_in = []
    for library in sorted(_package_dir().rglob("*.so")):
        listed = subprocess.run(
            [cuobjdump, "--list-ptx", str(library)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if newest in listed.replace(".", " ").split():
            found_in.append(library.name)

    assert found_in, (
        f"no shipped library carries portable device code for {newest}, the newest architecture "
        f"in this row ({sorted(expected)}). A GPU newer than {newest} would install this wheel and "
        "find no code it can run. The build appends the portable form for exactly this case, so "
        "either it was dropped or the spelling in the architecture list is wrong."
    )
    print(f"✓ portable device code for {newest} ships in {', '.join(found_in)}")


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


# Run in a child interpreter by test_a_model_runs_through_the_delegate, so the corrected library
# search path is in place before glibc caches it. Kept as source rather than a separate file
# because the smoke test is invoked directly and ships as one module.
_EXECUTION_CHILD = """
import os
import tempfile

import torch

from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


example = (torch.randn(4, 8), torch.randn(4, 8))
eager = Add()(*example)

exported = torch.export.export(Add().eval(), example)
lowered = to_edge_transform_and_lower(
    exported, partitioner=[CudaPartitioner([])]
).to_executorch()

# The program has to actually carry the delegate, or this passes while proving nothing about it.
assert b"CudaBackend" in lowered.buffer, (
    "the exported program contains no CUDA delegate, so running it would not exercise the "
    "shipped delegate at all"
)

# Loaded from the buffer rather than a file, and the output is brought back to host, because the
# delegate returns device memory and comparing it against an eager result on the host otherwise
# fails with an invalid argument rather than a mismatch.
module = _load_for_executorch_from_buffer(lowered.buffer)
actual = module.forward(list(example))[0]

torch.testing.assert_close(actual.cpu(), eager, rtol=1e-3, atol=1e-3)
capability = torch.cuda.get_device_capability(0)
print(f"PASS: a CUDA-delegated model ran on sm_{capability[0]}{capability[1]} and matched eager")


# The model above has no weights, so it needs nothing from the data file and runs whether or not
# the delegate's external data path works. Anything a user would actually ship has weights, and
# the CUDA delegate does not put them in the .pte: it writes them to a separate aoti_cuda_blob.ptd
# that the caller has to supply at load time. Nothing here loaded one, so a wheel that shipped a
# broken data path passed this check. Measured on sm_80: given the same model without the blob,
# the delegate reports the weights as not found, and the run then dies inside forward with an
# illegal memory access rather than returning wrong numbers.
class Weighted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


weighted = Weighted().eval()
weighted_example = (torch.randn(2, 8),)
with torch.no_grad():
    weighted_eager = weighted(*weighted_example)

weighted_program = to_edge_transform_and_lower(
    torch.export.export(weighted, weighted_example), partitioner=[CudaPartitioner([])]
).to_executorch()

assert b"CudaBackend" in weighted_program.buffer, (
    "the weighted program carries no CUDA delegate, so it would run on the portable kernels and "
    "never reach the external weights path this check exists for"
)

# Only the weight matrix, not every parameter. Measured on sm_80, the delegate wrote the 128-byte
# weight of this Linear into the data file and kept its 16-byte bias out, so the sum over
# parameters() is not a bound on what the file has to hold.
weight_bytes = weighted.fc.weight.numel() * weighted.fc.weight.element_size()

with tempfile.TemporaryDirectory() as work_dir:
    pte_path = os.path.join(work_dir, "weighted.pte")
    with open(pte_path, "wb") as handle:
        weighted_program.write_to_file(handle)
    weighted_program.write_tensor_data_to_file(work_dir)

    blob_path = os.path.join(work_dir, "aoti_cuda_blob.ptd")
    assert os.path.exists(blob_path), (
        "the export wrote no aoti_cuda_blob.ptd, so the weights of this model went nowhere and "
        f"every runner script that passes that file by name would fail: {sorted(os.listdir(work_dir))}"
    )

    # A data file is written even for a model with no weights, and it is not small: measured on
    # sm_80, the weightless model above produces a 256-byte container around an empty payload. So
    # the file's presence says nothing and its raw size says nothing either. What means something
    # is the amount by which this file exceeds that empty container, so write one and subtract it
    # instead of comparing against a constant, which would stop catching anything the next time
    # the container grows.
    empty_dir = os.path.join(work_dir, "weightless")
    os.mkdir(empty_dir)
    lowered.write_tensor_data_to_file(empty_dir)
    empty_size = os.path.getsize(os.path.join(empty_dir, "aoti_cuda_blob.ptd"))

    blob_size = os.path.getsize(blob_path)
    assert blob_size - empty_size >= weight_bytes, (
        f"aoti_cuda_blob.ptd is {blob_size} bytes against {empty_size} bytes for a model with no "
        f"weights, a difference of {blob_size - empty_size}, which does not cover this model's "
        f"{weight_bytes}-byte weight matrix, so the weights were not written to it"
    )

    with open(pte_path, "rb") as handle:
        pte_bytes = handle.read()
    with open(blob_path, "rb") as handle:
        blob_bytes = handle.read()

module = _load_for_executorch_from_buffer(pte_bytes, blob_bytes)
actual = module.forward(list(weighted_example))[0]

torch.testing.assert_close(actual.cpu(), weighted_eager, rtol=1e-3, atol=1e-3)
print(
    f"PASS: a CUDA-delegated model with weights ran on sm_{capability[0]}{capability[1]} from a "
    f"{blob_size}-byte aoti_cuda_blob.ptd carrying {blob_size - empty_size} bytes of weights, and "
    f"matched eager"
)
"""


def _cxx_runtime_dir() -> Optional[Path]:
    """The directory holding the C++ runtime this environment's compiler links against.

    The kernel library the delegate loads is compiled here, so it records this runtime's symbol
    versions. Returning the directory lets the caller put it where the loader will look.
    """
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_ENV")
    if not prefix:
        return None
    lib = Path(prefix) / "lib"
    return lib if (lib / "libstdc++.so.6").exists() else None


def _version_key(version: str) -> tuple:
    return tuple(int(part) for part in version.split("_")[1].split("."))


def _glibcxx_versions(library: Path) -> Set[str]:
    """The GLIBCXX version nodes a shared object defines or requires.

    Read with readelf rather than by parsing a filename, so a runtime that satisfies the kernel
    starts being used the moment it is present instead of when a list is updated by hand.
    """
    readelf = test_shared_libraries._tool("readelf")
    if readelf is None:
        return set()
    completed = subprocess.run(
        [readelf, "--version-info", "--wide", str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(re.findall(r"GLIBCXX_[0-9.]+", completed.stdout))


def test_a_model_runs_through_the_delegate() -> None:
    """Export two models to the CUDA delegate and run them, comparing against eager.

    Every other check in this file reads a shipped artifact. This one executes, because a wheel whose
    libraries are all present and correctly linked can still fail to compute, and nothing above would
    notice. The x86_64 rows land on a GPU runner, so this is the row where that can be proven.

    The second model carries weights. The CUDA delegate keeps those out of the .pte and writes them
    to a separate aoti_cuda_blob.ptd, which the caller then has to supply, so a wheel can run the
    weightless model and still be unable to run anything real. That path is only covered by the
    second model.

    Only the aarch64 rows may skip, and they say why, so a green result never stands for work that
    did not happen. On x86_64 the absent device or the mismatched torch build is itself the failure:
    that row is the one place execution is proven, and letting it report success for a check it did
    not run is how a delegate that cannot compute reaches a release.

    The body runs in a child interpreter. Export compiles the kernel library with this environment's
    compiler, so the library records that compiler's C++ runtime versions, while dlopen resolves
    libstdc++.so.6 from the loader's search path. Where the system copy is older than the compiler's
    the load fails on a missing version node. glibc caches the search list at startup, so the
    corrected path has to be in place before an interpreter begins, which is what the child gives.
    Only this check runs with it, because the checks above prove the shipped libraries resolve
    without a widened search path and would stop measuring that if it were widened for them too.
    """
    import torch

    execution_required = platform.machine() in ("x86_64", "amd64")

    if not torch.cuda.is_available():
        assert not execution_required, (
            "no CUDA device is visible to this x86_64 row, which is the row that proves the "
            "delegate computes. The runner lost its GPU or the installed torch cannot reach it; "
            "either way nothing here executed."
        )
        print(
            "SKIP: no CUDA device on this runner, so the delegate cannot execute here. "
            "This row still verifies the shipped libraries, their declared dependencies, "
            "their loader paths and the device code they carry."
        )
        return

    capability = torch.cuda.get_device_capability(0)
    device = f"sm_{capability[0]}{capability[1]}"
    if device not in torch.cuda.get_arch_list():
        assert not execution_required, (
            f"the torch resolved for this x86_64 row carries no code for {device}, the runner's "
            f"own GPU, so the execution check cannot run: {torch.cuda.get_arch_list()}"
        )
        print(
            f"SKIP: the installed torch carries no code for {device}, so nothing can run on this "
            f"device regardless of what the wheel ships."
        )
        return

    environment = dict(os.environ)
    runtime_dir = _cxx_runtime_dir()
    if runtime_dir is not None:
        search_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{runtime_dir}:{search_path}" if search_path else str(runtime_dir)
        )

    completed = subprocess.run(
        [sys.executable, "-c", _EXECUTION_CHILD],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        # The compiled kernel's own requirement against what the loader offered, so a version
        # mismatch reports the two numbers rather than only the load error the child saw.
        detail = ""
        if runtime_dir is not None:
            offered = _glibcxx_versions(runtime_dir / "libstdc++.so.6")
            if offered:
                detail = (
                    f" The runtime at {runtime_dir} offers up to "
                    f"{max(offered, key=_version_key)}."
                )
        raise AssertionError(
            f"a CUDA-delegated model did not run on {device}.{detail}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


if __name__ == "__main__":
    assert platform.system() == "Linux", "the CUDA rows are Linux only"

    test_cuda_libraries_are_shipped()
    test_cuda_runtime_is_declared()
    test_cuda_libraries_resolve_relatively()
    test_device_code_covers_the_row()
    test_portable_device_code_is_present()
    test_the_delegate_registers()
    test_a_model_runs_through_the_delegate()

    # The backend registrations a CPU Linux row asserts also apply here: the CUDA build enables
    # OpenVINO on every Linux architecture and downloads the QNN SDK on x86_64, so a CUDA wheel
    # carries both backends and needs both to register.
    from executorch.extension.pybindings.portable_lib import (
        _get_registered_backend_names,
    )

    registered = _get_registered_backend_names()
    if platform.machine() in ("x86_64", "amd64"):
        assert (
            "QnnBackend" in registered
        ), f"QnnBackend not found in registered backends: {registered}"
        print("✓ QnnBackend is registered")
    assert (
        "OpenvinoBackend" in registered
    ), f"OpenvinoBackend not found in registered backends: {registered}"
    print("✓ OpenvinoBackend is registered")

    test_base.test_cmsis_nn_install()

    # The packaging and linking checks a CPU wheel is held to still apply: one owner per
    # component, no build-tree paths, and a C++ application able to link what the wheel
    # ships.
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
