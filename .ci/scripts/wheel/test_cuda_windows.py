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
EXECUTORCH_CUDA_WINDOWS_ARTIFACTS at that directory here. Without it the check says so and
skips, the same way the Linux aarch64 rows skip execution, so a green result never stands
for work that did not happen.
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
    """The CUDA DLLs carry the CUDA runtime and need only the driver.

    CMake links the CUDA runtime statically by default on Windows, and nvidia publishes no
    CUDA runtime package for Windows, so the wheel declares none. What has to hold is that
    no CUDA DLL imports cudart64_*.dll, which a user without the CUDA toolkit on PATH could
    not satisfy. The runtime then reaches the driver, nvcuda.dll, which the display driver
    installs.
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
    ), f"the wheel declares {declared} for Windows, where the CUDA runtime is linked in"
    print(
        "✓ the CUDA runtime is linked in: no CUDA DLL imports cudart, only the driver"
    )


def _row_architectures() -> list:
    """The GPU architectures the row claims, as the build resolved them."""
    listed = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if listed:
        return [
            "sm_" + value.replace(".", "").replace("+PTX", "")
            for value in listed.split()
        ]
    return []


def test_device_code_covers_the_row() -> None:
    """Every GPU the row claims has device code in the shim library, which holds the kernels."""
    expected = _row_architectures()
    if not expected:
        assert os.environ.get("GITHUB_ACTIONS") != "true", (
            "a CUDA row reached the smoke test with no TORCH_CUDA_ARCH_LIST, so the build "
            "compiled device code only for whatever GPU the builder had"
        )
        print(
            "- no TORCH_CUDA_ARCH_LIST in this environment, skipping the device code check"
        )
        return
    cuobjdump = shutil.which("cuobjdump")
    assert cuobjdump, "cuobjdump from the CUDA toolkit is required to check device code"
    shims = _package_dir() / "backends" / "cuda" / "aoti_cuda_shims.dll"
    listed = subprocess.run(
        [cuobjdump, "--list-elf", str(shims)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    present = {
        token for token in listed.replace(".", " ").split() if token.startswith("sm_")
    }
    missing = sorted(set(expected) - present)
    assert not missing, (
        f"the row claims {expected} but {shims.name} carries no device code for {missing}. "
        f"Found: {sorted(present)}"
    )
    print(f"✓ device code in {shims.name} covers the row: {sorted(present)}")


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
    in the import table, that the package config copies the delegate's stream helper and
    shim layer beside the program, and that all of them load. Needs no GPU.
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
        f"C++ SDK:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
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
