#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
PYTORCH_ROCM_INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/test/rocm${ROCM_VERSION}}"
TORCHAO_ROCM_WHEEL_BASE="${TORCHAO_ROCM_WHEEL_BASE:-https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}}"
ROCM_CI_TMP_ROOT="${RUNNER_TEMP:-/tmp}"
mkdir -p "${ROCM_CI_TMP_ROOT}" 2>/dev/null || ROCM_CI_TMP_ROOT=/tmp
ROCM_CI_TMPDIR="$(mktemp -d "${ROCM_CI_TMP_ROOT}/executorch-rocm-ci.XXXXXX")"
trap 'rm -rf "${ROCM_CI_TMPDIR}"' EXIT

export ROCM_PATH
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="${ROCM_CI_TMPDIR}/inductor-cache"
export TORCHINDUCTOR_COMPILE_THREADS=1

read -r TORCH_VERSION TORCHAO_VERSION < <(
  python - <<'PY'
from install_requirements import TORCHAO_NIGHTLY_VERSION
from torch_pin import TORCH_VERSION

print(TORCH_VERSION, TORCHAO_NIGHTLY_VERSION)
PY
)
# TorchAO ROCm wheels are not exposed by the per-version pip index.
TORCHAO_WHEEL="${TORCHAO_ROCM_WHEEL_BASE}/torchao-${TORCHAO_VERSION}"
TORCHAO_WHEEL+="%2Brocm${ROCM_VERSION}-cp310-abi3-manylinux_2_28_x86_64.whl"
python -m pip install "torch==${TORCH_VERSION}" \
  --index-url "${PYTORCH_ROCM_INDEX}"
python -m pip install -r requirements-dev.txt \
  "${TORCHAO_WHEEL}"
EXECUTORCH_BUILD_MINIMAL=1 \
  python -m pip install --editable . --no-build-isolation

if ! command -v conda >/dev/null; then
  echo "The ROCm CI image must provide conda for its C++ runtime libraries"
  exit 1
fi
conda install -y -c conda-forge 'libstdcxx-ng>=12'

python - <<'PY'
import torch
import torchao
import triton

assert torch.version.hip is not None, "PyTorch is not a ROCm build"
assert torch.version.cuda is None, "PyTorch unexpectedly reports a CUDA runtime"
assert torch.cuda.is_available(), "No AMD GPU is visible through PyTorch"
assert "+rocm" in torchao.__version__, "TorchAO is not a ROCm build"

device = torch.cuda.get_device_properties(0)
arch = device.gcnArchName.split(":", 1)[0]
assert arch == "gfx950", f"Expected gfx950, got {device.gcnArchName}"
assert device.warp_size == 64, f"Expected wave64, got {device.warp_size}"
assert arch in torch.cuda.get_arch_list(), (
    f"{arch} is not supported by this PyTorch build: {torch.cuda.get_arch_list()}"
)

print("PyTorch:", torch.__version__)
print("TorchAO:", torchao.__version__)
print("ROCm:", torch.version.hip)
print("Triton:", triton.__version__)
print("Device:", device.name, device.gcnArchName)
print("Architectures:", torch.cuda.get_arch_list())
PY

if command -v rocminfo >/dev/null; then
  rocminfo | sed -n '1,160p'
fi
if command -v rocm-smi >/dev/null; then
  rocm-smi --showproductname --showmeminfo vram --showuse --showtemp || true
elif command -v amd-smi >/dev/null; then
  amd-smi static --gpu all || true
fi

TORCH_CMAKE_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}"

cmake --preset llm-release-rocm -DEXECUTORCH_BUILD_TESTS=ON
cmake --build cmake-out-rocm-llm \
  --target executor_runner test_cuda_allocator test_cuda_mutable_state \
  --parallel "$(nproc)"

PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${PWD}/cmake-out-rocm-llm/backends/cuda:${PWD}/cmake-out-rocm-llm/extension/cuda:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

ctest --test-dir cmake-out-rocm-llm \
  -R 'test_cuda_(allocator|mutable_state)' \
  -V

# Targeted builds leave the shim in the build tree rather than install lib/.
for ROCM_CI_BINARY in \
  cmake-out-rocm-llm/backends/cuda/libaoti_cuda_shims.so \
  cmake-out-rocm-llm/executor_runner; do
  ROCM_CI_LINKS="$(ldd "${ROCM_CI_BINARY}")"
  printf '%s\n' "${ROCM_CI_LINKS}"
  if grep -q 'not found' <<<"${ROCM_CI_LINKS}"; then
    echo "Missing runtime dependency in ${ROCM_CI_BINARY}"
    exit 1
  fi
  if grep -q 'libcudart' <<<"${ROCM_CI_LINKS}"; then
    echo "Unexpected CUDA runtime dependency in ${ROCM_CI_BINARY}"
    exit 1
  fi
  if ! grep -q 'libamdhip64' <<<"${ROCM_CI_LINKS}"; then
    echo "HIP runtime dependency missing from ${ROCM_CI_BINARY}"
    exit 1
  fi
done

if ! grep -q '^EXECUTORCH_BUILD_CUDA:BOOL=OFF$' \
  cmake-out-rocm-llm/CMakeCache.txt; then
  echo "ROCm CI unexpectedly enabled the CUDA build"
  exit 1
fi
if ! grep -q '^EXECUTORCH_BUILD_ROCM:BOOL=ON$' \
  cmake-out-rocm-llm/CMakeCache.txt; then
  echo "ROCm CI did not enable the ROCm build"
  exit 1
fi
if ! grep -q 'find_dependency(hip CONFIG)' \
  cmake-out-rocm-llm/executorch-backend-dependencies.cmake; then
  echo "ROCm package metadata does not declare its HIP dependency"
  exit 1
fi
if grep -q 'find_dependency(CUDAToolkit)' \
  cmake-out-rocm-llm/executorch-backend-dependencies.cmake; then
  echo "ROCm package metadata unexpectedly depends on CUDAToolkit"
  exit 1
fi

export EXECUTORCH_EXECUTOR_RUNNER="${PWD}/cmake-out-rocm-llm/executor_runner"
ROCM_EXCLUDED_TESTS=(
  # This focused job does not install the optional flash-linear-attention package.
  --ignore=backends/cuda/tests/test_chunk_gated_delta_rule.py
  # Defer specialized SDPA suites until their CI cost is measured.
  --ignore=backends/cuda/tests/test_tq4_sdpa.py
  --ignore=backends/cuda/tests/test_triton_sdpa.py
  --ignore=backends/cuda/tests/test_triton_sdpa_splitk.py
  # This test process core-dumps on ROCm instead of reporting a failure.
  --ignore=backends/cuda/tests/test_triton_sdpa_nan.py
)
python -m pytest -v -o 'addopts=' \
  "${ROCM_EXCLUDED_TESTS[@]}" \
  backends/cuda/tests \
  backends/cuda/passes/tests

ROCM_POINTWISE_DIR="${ROCM_CI_TMPDIR}/pointwise"
python -m examples.cuda.scripts.export_amd_pointwise \
  --output-dir "${ROCM_POINTWISE_DIR}"
