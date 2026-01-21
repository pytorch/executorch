# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs

# Unlike the rest of PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python util/collect_env.py` or `python -m util.collect_env`

import datetime
import json
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "pip_version",
        "pip_packages",
        "conda_packages",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
    ],
)

COMMON_PATTERNS = ["torch", "numpy", "triton", "optree"]

NVIDIA_PATTERNS = [
    "cuda",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvtx",
]

CONDA_PATTERNS = ["cudatoolkit", "mkl", "magma"]

PIP_PATTERNS = ["mypy", "flake8", "onnx"]


def run(command):
    shell = isinstance(command, str)
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    out, err = p.communicate()
    enc = "oem" if get_platform() == "win32" else locale.getpreferredencoding()
    return p.returncode, out.decode(enc).strip(), err.decode(enc).strip()


def run_and_read_all(run_lambda, command):
    rc, out, _ = run_lambda(command)
    return out if rc == 0 else None


def run_and_parse_first_match(run_lambda, command, regex):
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    m = re.search(regex, out)
    return m.group(1) if m else None


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "clang --version", r"clang version (.*)")


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_smi():
    smi = "nvidia-smi"
    if get_platform() == "win32":
        for p in [
            os.path.join(os.environ.get("SYSTEMROOT", ""), "System32", smi),
            os.path.join(
                os.environ.get("PROGRAMFILES", ""),
                "NVIDIA Corporation",
                "NVSMI",
                smi,
            ),
        ]:
            if os.path.exists(p):
                return f'"{p}"'
    return smi


def get_nvidia_driver_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, get_nvidia_smi(), r"Driver Version: (.*?) "
    )


def get_gpu_info(run_lambda):
    rc, out, _ = run_lambda(get_nvidia_smi() + " -L")
    return out if rc == 0 else None


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"V(\d+\.\d+)")


def get_pip_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = PIP_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS

    pip_version = "pip3" if sys.version_info.major == 3 else "pip"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    out = run_and_read_all(
        run_lambda, [sys.executable, "-mpip", "list", "--format=freeze"]
    )

    # ✅ FIX: prevent crash if pip fails
    if out is None:
        return pip_version, ""

    filtered = "\n".join(
        line for line in out.splitlines() if any(p in line for p in patterns)
    )
    return pip_version, filtered


def get_conda_packages(run_lambda):
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, f"{conda} list")
    if out is None:
        return None
    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#")
        and any(p in line for p in COMMON_PATTERNS + NVIDIA_PATTERNS)
    )


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def get_os(run_lambda):
    if get_platform() == "win32":
        return run_and_read_all(run_lambda, "ver")
    if get_platform() == "darwin":
        return run_and_read_all(run_lambda, "sw_vers")
    return run_and_read_all(run_lambda, "uname -a")


def get_python_platform():
    import platform

    return platform.platform()


def get_libc_version():
    if get_platform() != "linux":
        return "N/A"
    import platform

    return "-".join(platform.libc_ver())


def get_env_info():
    run_lambda = run
    pip_version, pip_packages = get_pip_packages(run_lambda)
    conda_packages = get_conda_packages(run_lambda)

    return SystemEnv(
        torch_version=getattr(torch, "__version__", "N/A") if TORCH_AVAILABLE else "N/A",
        is_debug_build=str(getattr(torch.version, "debug", "N/A"))
        if TORCH_AVAILABLE
        else "N/A",
        cuda_compiled_version=getattr(torch.version, "cuda", "N/A")
        if TORCH_AVAILABLE
        else "N/A",
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        python_version=sys.version.replace("\n", " "),
        python_platform=get_python_platform(),
        is_cuda_available=str(torch.cuda.is_available()) if TORCH_AVAILABLE else "N/A",
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=os.environ.get("CUDA_MODULE_LOADING", ""),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        cudnn_version=None,
        pip_version=pip_version,
        pip_packages=pip_packages,
        conda_packages=conda_packages,
        hip_compiled_version="N/A",
        hip_runtime_version="N/A",
        miopen_runtime_version="N/A",
        caching_allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        is_xnnpack_available=str(
            getattr(torch.backends, "xnnpack", None) is not None
        ),
        cpu_info=None,
    )


def pretty_str(envinfo):
    return "\n".join(f"{k}: {v}" for k, v in envinfo._asdict().items())


def get_pretty_env_info():
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    print(get_pretty_env_info())


if __name__ == "__main__":
    main()
