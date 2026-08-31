#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.1}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
EXPECTED_ROCM_ARCH="${EXPECTED_ROCM_ARCH:-gfx950}"
EXPECTED_WARP_SIZE="${EXPECTED_WARP_SIZE:-64}"
PYTORCH_ROCM_INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/test/rocm${ROCM_VERSION}}"
TORCHAO_ROCM_WHEEL_BASE="${TORCHAO_ROCM_WHEEL_BASE:-https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}}"
VOXTRAL_CI_TMP_ROOT="${RUNNER_TEMP:-/tmp}"
if ! mkdir -p "${VOXTRAL_CI_TMP_ROOT}" 2>/dev/null ||
  [[ ! -w "${VOXTRAL_CI_TMP_ROOT}" ]]; then
  VOXTRAL_CI_TMP_ROOT=/tmp
fi
VOXTRAL_CI_TMPDIR="$(mktemp -d "${VOXTRAL_CI_TMP_ROOT}/executorch-rocm-voxtral.XXXXXX")"
trap 'rm -rf "${VOXTRAL_CI_TMPDIR}"' EXIT
VOXTRAL_START_SECONDS="${SECONDS}"

export ROCM_PATH
export EXPECTED_ROCM_ARCH
export EXPECTED_WARP_SIZE
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${VOXTRAL_CI_TMP_ROOT}/hf-cache"
export HF_HUB_DISABLE_XET=1
mkdir -p "${HF_HOME}"
HF_TOKEN="$(printf '%s' "${SECRET_EXECUTORCH_HF_TOKEN}" | tr -d '\r\n')"
export HF_TOKEN

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
python -m pip install --editable . --no-build-isolation

if ! command -v conda >/dev/null; then
  echo "The ROCm CI image must provide conda for its runtime libraries"
  exit 1
fi
conda install -y -c conda-forge ffmpeg 'libstdcxx-ng>=12'

python -m pip install 'fsspec[http]<=2025.3.0'
python -m pip install datasets huggingface_hub librosa mistral-common safetensors soundfile
python -m pip install torchcodec==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/test/cpu

python - <<'PY'
import os

import torch
import torchao

assert torch.version.hip is not None, "PyTorch is not a ROCm build"
assert torch.version.cuda is None, "PyTorch unexpectedly reports a CUDA runtime"
assert torch.cuda.is_available(), "No AMD GPU is visible through PyTorch"
assert "+rocm" in torchao.__version__, "TorchAO is not a ROCm build"

device = torch.cuda.get_device_properties(0)
arch = device.gcnArchName.split(":", 1)[0]
expected_arch = os.environ["EXPECTED_ROCM_ARCH"]
expected_warp_size = int(os.environ["EXPECTED_WARP_SIZE"])
assert arch == expected_arch, f"Expected {expected_arch}, got {device.gcnArchName}"
assert device.warp_size == expected_warp_size, (
    f"Expected wave{expected_warp_size}, got {device.warp_size}"
)
print(
    torch.__version__,
    torchao.__version__,
    torch.version.hip,
    device.name,
    device.gcnArchName,
)
PY

python -m pytest -v -o 'addopts=' examples/models/voxtral_realtime/tests

VOXTRAL_MODEL_DIR="${VOXTRAL_CI_TMPDIR}/model"
VOXTRAL_AUDIO="${VOXTRAL_CI_TMPDIR}/test_audio.wav"
VOXTRAL_OUTPUT="${VOXTRAL_CI_TMPDIR}/output"
VOXTRAL_LOG="${VOXTRAL_CI_TMPDIR}/voxtral.log"

python - "${VOXTRAL_MODEL_DIR}" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    local_dir=sys.argv[1],
    allow_patterns=["params.json", "tekken.json", "consolidated.safetensors"],
)
PY

python - "${VOXTRAL_AUDIO}" <<'PY'
import sys

import soundfile as sf
from datasets import load_dataset

sample = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")[0][
    "audio"
]
sf.write(sys.argv[1], sample["array"][: sample["sampling_rate"] * 30], sample["sampling_rate"])
PY

CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH
VOXTRAL_PYTHON=python \
  bash examples/models/voxtral_realtime/run_rocm_e2e.sh \
    "${VOXTRAL_MODEL_DIR}" \
    "${VOXTRAL_AUDIO}" \
    w4-bf16 \
    streaming \
    "${VOXTRAL_OUTPUT}" 2>&1 | tee "${VOXTRAL_LOG}"

if ! grep -qi 'Quilter' "${VOXTRAL_LOG}"; then
  echo "Voxtral transcript did not contain the expected word: Quilter"
  exit 1
fi
echo "Voxtral transcript contains the expected word: Quilter"

for VOXTRAL_METRIC in 'Voxtral export metrics:' 'Voxtral RTF:'; do
  if ! grep -q "${VOXTRAL_METRIC}" "${VOXTRAL_LOG}"; then
    echo "Voxtral CI did not report ${VOXTRAL_METRIC}"
    exit 1
  fi
done

du -sh "${VOXTRAL_MODEL_DIR}" "${VOXTRAL_OUTPUT}"
echo "Voxtral ROCm CI wall time: $((SECONDS - VOXTRAL_START_SECONDS)) seconds"
