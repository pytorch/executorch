#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for a Windows CUDA wheel row.

The Windows CUDA wheel ships the CUDA delegate for C++ applications only. A CUDA program
cannot be lowered on Windows, since lowering compiles the model with a toolchain that only
the Linux side has, so a Windows user exports on Linux (or WSL) and runs the program with
this wheel. The Python extension therefore carries no CUDA dependency at all, and the
checks here hold both halves of that:

  the CUDA DLLs ship, with import libraries a C++ consumer links
  the Python extension depends on no CUDA DLL, so importing the package needs no CUDA
  the delegate registers in a C++ application linking executorch::backend_cuda
  the device code covers the GPUs the row claims
  a program exported on Linux for a Windows target runs through the C++ SDK and matches eager

The last check needs such a program. Run this file with --export DIR on a Linux machine
with CUDA and the Windows cross toolchain to write one, then point
EXECUTORCH_CUDA_WINDOWS_ARTIFACTS at that directory here. CI does exactly that in the
Windows CUDA wheel workflow's end-to-end jobs, one per CUDA train. The wheel build's own
smoke test has no exported program, so there the check prints SKIP and only the checks
that need no GPU and no program count.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_EXPORT_SCRIPT = """
import json
import sys
from pathlib import Path

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec


class Net(torch.nn.Module):
    # Weights, so the program has an external data file, and several operator kinds.
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x))) + x[:, :8]


destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
model = Net().eval()
example = (torch.randn(4, 16),)
with torch.no_grad():
    expected = model(*example)

compile_specs = [
    CudaBackend.generate_method_name_compile_spec("forward"),
    CompileSpec("platform", b"windows"),
]
program = to_edge_transform_and_lower(
    torch.export.export(model, example),
    partitioner=[CudaPartitioner(compile_specs)],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
).to_executorch()
assert b"CudaBackend" in program.buffer, "the program carries no CUDA delegate"
with open(destination / "model.pte", "wb") as handle:
    program.write_to_file(handle)
program.write_tensor_data_to_file(str(destination))
assert (destination / "aoti_cuda_blob.ptd").is_file(), "no aoti_cuda_blob.ptd was written"
(destination / "reference.json").write_text(
    json.dumps(
        {
            "shape": list(example[0].shape),
            "input": example[0].flatten().tolist(),
            "expected": expected.flatten().tolist(),
            # The CUDA train the program was compiled with, which the Windows side
            # checks against the wheel's, since every CUDA 13 train loads the same
            # cudart64_13.dll and a mismatch would not fail on its own.
            "cuda": torch.version.cuda,
        }
    )
)
print(f"exported a Windows CUDA program to {destination}")
"""

_REGISTRY_SOURCE = r"""
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>

int main() {
  executorch::runtime::runtime_init();
  const size_t count = executorch::runtime::get_num_registered_backends();
  for (size_t i = 0; i < count; ++i) {
    const auto name = executorch::runtime::get_backend_name(i);
    if (name.ok()) {
      std::printf("BACKEND %s\n", *name);
    }
  }
  return 0;
}
"""

_RUNNER_SOURCE = r"""
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace executorch::extension;

std::vector<float> read_floats(const char* path) {
  std::ifstream file(path);
  std::vector<float> values;
  float value = 0.0f;
  while (file >> value) {
    values.push_back(value);
  }
  return values;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::printf("usage: runner <pte> <ptd> <rows> <cols> <input> <expected>\n");
    return 2;
  }
  Module module(argv[1], argv[2]);
  auto input_data = read_floats(argv[5]);
  const auto expected = read_floats(argv[6]);
  auto input = make_tensor_ptr(
      {std::atoi(argv[3]), std::atoi(argv[4])}, std::move(input_data));
  const auto result = module.forward(input);
  if (!result.ok()) {
    std::printf("forward failed: 0x%x\n", static_cast<unsigned>(result.error()));
    return 1;
  }
  const auto output = result->at(0).toTensor();
  if (static_cast<size_t>(output.numel()) != expected.size()) {
    std::printf("output has %zu values, expected %zu\n",
                static_cast<size_t>(output.numel()), expected.size());
    return 1;
  }
  const float* actual = output.const_data_ptr<float>();
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(actual[i]) - expected[i]);
    if (!std::isfinite(diff)) {
      std::printf("output value %zu is not comparable\n", i);
      return 1;
    }
    worst = diff > worst ? diff : worst;
  }
  if (worst > 1e-3) {
    std::printf("output differs from eager PyTorch by %g\n", worst);
    return 1;
  }
  std::printf("ok maxdiff=%g\n", worst);
  return 0;
}
"""


def _package_dir() -> Path:
    import executorch

    return Path(executorch.__path__[0])


def test_cuda_libraries_are_shipped() -> None:
    """The row is named for CUDA, so the delegate, its helpers and their link inputs ship."""
    package = _package_dir()
    expected = [
        package / "lib" / "executorch_backend_cuda.dll",
        package / "lib" / "executorch_backend_cuda.lib",
        package / "lib" / "executorch_extension_cuda.dll",
        package / "lib" / "executorch_extension_cuda.lib",
        package / "backends" / "cuda" / "aoti_cuda_shims.dll",
        # What a compiled program links when lowering for Windows, and what the package
        # config names as the shim layer's import library.
        package / "data" / "lib" / "aoti_cuda_shims.lib",
    ]
    missing = [
        str(path.relative_to(package)) for path in expected if not path.is_file()
    ]
    assert not missing, (
        f"this is a CUDA row but {missing} are not in the wheel, so a C++ application could "
        "not link or load the CUDA delegate"
    )
    print(f"✓ the CUDA DLLs and their import libraries ship ({len(expected)} files)")


def test_python_extension_carries_no_cuda() -> None:
    """The Python extension must depend on no CUDA DLL.

    CUDA programs are lowered on Linux, so the Windows extension has nothing to do with the
    delegate, and depending on it would make importing the package fail on a machine
    without the CUDA runtime.
    """
    import test_shared_libraries

    package = _package_dir()
    extensions = sorted((package / "extension" / "pybindings").glob("_C.*.pyd"))
    assert len(extensions) == 1, f"expected one _C extension, found {extensions}"
    dependents = {
        name.lower() for name in test_shared_libraries._pe_dependents(extensions[0])
    }
    cuda = sorted(
        name
        for name in dependents
        if "cuda" in name or name.startswith(("cudart", "cublas", "nvrtc"))
    )
    assert not cuda, (
        f"{extensions[0].name} depends on {cuda}, so the Python side of the Windows wheel "
        "carries CUDA it cannot use and fails to import without the CUDA runtime"
    )
    from executorch.extension.pybindings.portable_lib import (
        _get_registered_backend_names,
    )

    registered = _get_registered_backend_names()
    assert "CudaBackend" not in registered, (
        f"CudaBackend is registered in the Python extension ({registered}), which the "
        "Windows wheel does not ship it for"
    )
    print(
        f"✓ {extensions[0].name} depends on no CUDA DLL and registers no CUDA backend"
    )


def test_cuda_runtime_is_linked_statically() -> None:
    """The wheel's CUDA DLLs import no CUDA runtime DLL, only reaching the driver.

    On Windows CMake links the CUDA runtime library (cudart.lib) into each DLL, and that
    library loads the driver, nvcuda.dll, which the display driver installs. So no shipped
    DLL may import cudart64_*.dll.

    A compiled model is different: the library AOTInductor builds into the .pte imports
    cudart64_13.dll, which comes from the CUDA Toolkit's bin directory on PATH. The
    nvidia-cuda-runtime package on PyPI also carries it for win_amd64, but a C++ program
    does not search site-packages for DLLs, so declaring it would not make it loadable; the
    wheel declares no NVIDIA package for Windows and the guide says where the DLL comes from.
    """
    import test_shared_libraries

    package = _package_dir()
    for library in (
        package / "lib" / "executorch_backend_cuda.dll",
        package / "lib" / "executorch_extension_cuda.dll",
        package / "backends" / "cuda" / "aoti_cuda_shims.dll",
    ):
        dependents = {
            name.lower() for name in test_shared_libraries._pe_dependents(library)
        }
        cudart = sorted(name for name in dependents if name.startswith("cudart"))
        assert not cudart, (
            f"{library.name} imports {cudart}, which only the CUDA toolkit provides on "
            "Windows, so the wheel would fail to load without it"
        )
    shims = package / "backends" / "cuda" / "aoti_cuda_shims.dll"
    assert b"nvcuda.dll" in shims.read_bytes(), (
        f"{shims.name} carries no reference to the CUDA driver, so the CUDA runtime it "
        "should contain is missing"
    )
    import importlib.metadata

    declared = [
        requirement
        for requirement in importlib.metadata.requires("executorch") or []
        if "nvidia" in requirement.lower() and "linux" not in requirement.lower()
    ]
    assert (
        not declared
    ), f"the wheel declares {declared} for Windows, where its DLLs need only the driver"
    print("✓ no shipped CUDA DLL imports a CUDA runtime DLL, only the driver")


def _row_architectures() -> list:
    """The GPU architectures this row claims, from the list the build is meant to use.

    From executorch_cuda_arch_list in cuda_arch_list.sh, called the way the Linux check and
    the pre-build hook call it, keyed by the wheel's own +cuXYZ version. Not from
    TORCH_CUDA_ARCH_LIST: that is what the build consumed, so a check that trusted it would
    pass a build that was handed the wrong list, which is exactly what happened once (the
    reusable workflow's list replaced the row's, adding sm_75 and dropping sm_89).
    """
    import importlib.metadata

    local = importlib.metadata.version("executorch").partition("+")[2]
    assert re.fullmatch(
        r"cu\d+", local
    ), f"this is run as a CUDA row but the installed version carries no +cuXYZ tag: {local!r}"
    return ["sm_" + value.replace(".", "") for value in _arch_list(local).split()]


def _bash() -> str:
    """Git's bash where it is installed, as on the runners, else whatever bash is on PATH.

    System32's bash.exe is WSL's launcher, which can come first on PATH and fails on a
    machine with no WSL distribution.
    """
    git_bash = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
    )
    return str(git_bash) if git_bash.is_file() else "bash"


def _arch_list(train: str) -> str:
    """executorch_cuda_arch_list for an x86_64 row of `train`, as the release build gets it.

    bash is Git's on the Windows runners and WSL's launcher on a developer machine, which
    shares neither the environment nor absolute paths, so the train goes on the command line
    and the script is read relative to the directory bash starts in, without carriage
    returns, which a checkout with core.autocrlf adds and bash cannot parse.
    """
    listed = subprocess.run(
        [
            _bash(),
            "-c",
            f"CU_VERSION={train}; source <(tr -d '\\r' < ./cuda_arch_list.sh) "
            "&& executorch_cuda_arch_list",
        ],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert listed.returncode == 0 and listed.stdout.split(), (
        f"cuda_arch_list.sh gives no GPU list for {train}: "
        f"{(listed.stdout + listed.stderr).strip()}"
    )
    return listed.stdout.strip()


def _device_code(cuobjdump: str, library: Path, kind: str) -> set:
    """The sm_XY architectures cuobjdump lists for a library, as ELF ("elf") or PTX ("ptx")."""
    listed = subprocess.run(
        [cuobjdump, f"--list-{kind}", str(library)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {
        token for token in listed.replace(".", " ").split() if token.startswith("sm_")
    }


def test_device_code_covers_the_row() -> None:
    """Every DLL with device code carries exactly the row's GPUs, and the newest as PTX too.

    Both directions, per library: a missing architecture is a GPU the row claims and cannot
    run on, and an extra one means the build did not use the row's list. The newest
    architecture must also ship in its portable form, so a GPU newer than any in the row can
    compile it on load. --list-elf cannot see that form; --list-ptx can.
    """
    expected = set(_row_architectures())
    cuobjdump = shutil.which("cuobjdump")
    assert cuobjdump, "cuobjdump from the CUDA toolkit is required to check device code"
    package = _package_dir()
    with_device_code = {}
    for library in sorted(package.rglob("*.dll")):
        found = _device_code(cuobjdump, library, "elf")
        if found:
            with_device_code[library] = found
    assert (
        with_device_code
    ), f"no shipped DLL carries GPU device code, while the row claims {sorted(expected)}"
    for library, found in with_device_code.items():
        missing, extra = sorted(expected - found), sorted(found - expected)
        assert not missing and not extra, (
            f"{library.name} carries device code for {sorted(found)} but the row claims "
            f"{sorted(expected)}: missing {missing}, unexpected {extra}"
        )
    newest = max(expected, key=lambda arch: int(arch.removeprefix("sm_")))
    without_portable = [
        library.name
        for library in with_device_code
        if newest not in _device_code(cuobjdump, library, "ptx")
    ]
    assert not without_portable, (
        f"{', '.join(without_portable)} carry device code but no portable form of {newest}, "
        "the newest architecture in the row, so a newer GPU would find no code it can run"
    )
    print(
        f"✓ device code covers exactly the row {sorted(expected)} in "
        f"{', '.join(library.name for library in with_device_code)}, with {newest} also as PTX"
    )


def _build(work_dir: Path, name: str, source: str, components) -> Path:
    import test_cpp_sdk

    source_dir = work_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(source)
    (source_dir / "CMakeLists.txt").write_text(test_cpp_sdk._consumer_cmake(components))
    config = _package_dir() / "share" / "cmake"
    build_dir = work_dir / f"{name}-build"
    for command in (
        [
            test_cpp_sdk._tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config}",
        ],
        test_cpp_sdk._cmake_build(test_cpp_sdk._tool("cmake"), build_dir),
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (
            f"a C++ application linking {components} could not be built against the "
            f"installed wheel:\n{result.stdout[-2500:]}\n{result.stderr[-2500:]}"
        )
    return test_cpp_sdk._executable(build_dir, "consumer")


def test_the_delegate_registers_in_a_cpp_application(work_dir: Path) -> None:
    """A C++ application linking executorch::backend_cuda sees CudaBackend registered.

    The delegate registers from a static initializer, so this proves the anchor keeps the DLL
    in the import table, that the package config copies the delegate's CUDA extension
    (allocator and stream helpers) and shim layer beside the program, and that all of them
    load. Needs no GPU.
    """
    import test_cpp_sdk

    consumer = _build(
        work_dir,
        "cuda-registry",
        _REGISTRY_SOURCE,
        ["runtime", "kernels_optimized", "backend_cuda"],
    )
    beside = {path.name for path in consumer.parent.glob("*.dll")}
    for needed in (
        "executorch_backend_cuda.dll",
        "executorch_extension_cuda.dll",
        "aoti_cuda_shims.dll",
    ):
        assert needed in beside, (
            f"$<TARGET_RUNTIME_DLLS> did not copy {needed} beside the application, so it "
            f"cannot load the delegate. Copied: {sorted(beside)}"
        )
    result = subprocess.run(
        [str(consumer)],
        capture_output=True,
        text=True,
        check=False,
        env=test_cpp_sdk._loader_clean_environment(),
        timeout=test_cpp_sdk._RUN_TIMEOUT,
    )
    assert result.returncode == 0, (
        "a C++ application linking the CUDA delegate failed to start, so a DLL it needs is "
        f"missing:\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}"
    )
    backends = re.findall(r"^BACKEND (\S+)", result.stdout, re.M)
    assert "CudaBackend" in backends, (
        f"the application linked executorch::backend_cuda but CudaBackend is not registered: "
        f"{backends}"
    )
    print(f"✓ a C++ application linking executorch::backend_cuda registers {backends}")


def _cuda_environment() -> str:
    """The driver and runtime this machine offers, for reading a failure that happens there.

    On Windows every CUDA 13 runtime library, the toolkit's cudart64_13.dll included, loads
    the runtime that ships with the display driver (nvcudart_hybrid64.dll), so whether a
    program runs depends on the driver more than on the toolkit. CI runners do not put
    nvidia-smi on PATH, so this asks the runtime directly.
    """
    import ctypes

    lines = []
    nvcuda = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvcuda.dll"
    )
    lines.append(f"nvcuda.dll present: {nvcuda.is_file()}")
    smi = shutil.which("nvidia-smi") or str(nvcuda.parent / "nvidia-smi.exe")
    if Path(smi).is_file():
        query = subprocess.run(
            [
                smi,
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        lines.append(f"nvidia-smi: {(query.stdout or query.stderr).strip()}")
    runtime = next(
        (
            candidate
            for directory in os.environ.get("PATH", "").split(os.pathsep)
            if directory
            for candidate in sorted(Path(directory).glob("cudart64_*.dll"))
        ),
        None,
    )
    if runtime is None:
        lines.append("no cudart64_*.dll on PATH to ask")
        return "\n".join(lines)
    try:
        library = ctypes.WinDLL(str(runtime))
    except OSError as error:
        lines.append(f"{runtime} does not load: {error}")
        return "\n".join(lines)
    library.cudaGetErrorName.restype = ctypes.c_char_p
    driver, version, count = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    library.cudaDriverGetVersion(ctypes.byref(driver))
    library.cudaRuntimeGetVersion(ctypes.byref(version))
    status = library.cudaGetDeviceCount(ctypes.byref(count))
    name = library.cudaGetErrorName(status)
    lines.append(
        f"{runtime.name}: driver supports CUDA {driver.value}, runtime {version.value}, "
        f"cudaGetDeviceCount -> {name.decode() if name else status} ({count.value} devices)"
    )
    return "\n".join(lines)


def test_a_program_exported_on_linux_runs(work_dir: Path) -> None:
    """A CUDA program lowered on Linux for Windows runs through the C++ SDK and matches eager.

    This is the only route a Windows user has, and the only check here that computes. It needs
    artifacts written by `test_cuda_windows.py --export DIR` on Linux and a CUDA device.
    """
    import test_cpp_sdk

    artifacts = os.environ.get("EXECUTORCH_CUDA_WINDOWS_ARTIFACTS", "")
    if not artifacts:
        print(
            "SKIP: EXECUTORCH_CUDA_WINDOWS_ARTIFACTS is not set, so no Linux-exported program "
            "is available to run. Write one with `test_cuda_windows.py --export DIR` on Linux."
        )
        return
    artifacts = Path(artifacts)
    reference = json.loads((artifacts / "reference.json").read_text())
    import importlib.metadata

    wheel_train = importlib.metadata.version("executorch").partition("+")[2]
    exported_train = "cu" + str(reference.get("cuda", "")).replace(".", "")
    assert exported_train == wheel_train, (
        f"the program was exported with CUDA {reference.get('cuda')} but this wheel is the "
        f"{wheel_train} build, so this would not test the pairing the row claims"
    )
    input_file = work_dir / "cuda_input.data"
    expected_file = work_dir / "cuda_expected.data"
    input_file.write_text(" ".join(repr(v) for v in reference["input"]))
    expected_file.write_text(" ".join(repr(v) for v in reference["expected"]))
    runner = _build(
        work_dir,
        "cuda-runner",
        _RUNNER_SOURCE,
        ["runtime", "kernels_optimized", "backend_cuda"],
    )
    # Only context for reading a failure, so it must not be what fails: an nvidia-smi
    # that hangs or does not start would otherwise stop the check before the program runs.
    try:
        environment = _cuda_environment()
    except (OSError, subprocess.SubprocessError) as error:
        environment = f"could not query the CUDA environment: {error}"
    print(environment)
    result = subprocess.run(
        [
            str(runner),
            str(artifacts / "model.pte"),
            str(artifacts / "aoti_cuda_blob.ptd"),
            str(reference["shape"][0]),
            str(reference["shape"][1]),
            str(input_file),
            str(expected_file),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=test_cpp_sdk._loader_clean_environment(),
        timeout=test_cpp_sdk._RUN_TIMEOUT,
    )
    assert result.returncode == 0, (
        "a CUDA program exported on Linux did not run correctly through the Windows wheel's "
        f"C++ SDK:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}\n"
        f"CUDA on this machine:\n{environment}"
    )
    print(
        f"✓ a Linux-exported CUDA program runs on Windows and matches eager ({result.stdout.strip().splitlines()[-1]})"
    )


def export(destination: str) -> None:
    """Write a Windows-targeted CUDA program for the execution check. Run on Linux."""
    assert platform.system() == "Linux", "lowering for Windows runs on Linux"
    subprocess.run([sys.executable, "-c", _EXPORT_SCRIPT, destination], check=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export":
        export(sys.argv[2])
        sys.exit(0)

    assert platform.system() == "Windows", "this is the Windows CUDA row's smoke test"
    import test_clean_install
    import test_cpp_sdk
    import test_shared_libraries

    with tempfile.TemporaryDirectory() as work_dir:
        test_clean_install.run_tests(Path(work_dir))

    test_cuda_libraries_are_shipped()
    test_python_extension_carries_no_cuda()
    test_cuda_runtime_is_linked_statically()
    test_device_code_covers_the_row()
    with tempfile.TemporaryDirectory() as work_dir:
        test_the_delegate_registers_in_a_cpp_application(Path(work_dir))
    with tempfile.TemporaryDirectory() as work_dir:
        test_a_program_exported_on_linux_runs(Path(work_dir))

    # Everything a CPU Windows row checks applies here too.
    with tempfile.TemporaryDirectory() as work_dir:
        test_shared_libraries.run_tests(Path(work_dir))
    with tempfile.TemporaryDirectory() as work_dir:
        test_cpp_sdk.run_tests(Path(work_dir))
    import test_base

    test_base.test_cmsis_nn_install()
    # The same model run the CPU row ends with, through the Python bindings.
    import test_windows
    from executorch.examples.models import Backend, Model
    from test_base import ModelTest

    test_windows.run_tests(
        model_tests=[ModelTest(model=Model.Mv3, backend=Backend.Xnnpack)]
    )
