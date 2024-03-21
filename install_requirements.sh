#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install required python dependencies for developing
# Dependencies are defined in .pyproject.toml
if [[ -z $BUCK ]];
then
  BUCK=buck2
fi

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

# Parse options.
EXECUTORCH_BUILD_PYBIND=OFF
CMAKE_ARGS=""

for arg in "$@"; do
  case $arg in
    --pybind)
      EXECUTORCH_BUILD_PYBIND=ON
      ;;
    coreml|mps|xnnpack)
      if [[ "$EXECUTORCH_BUILD_PYBIND" == "ON" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DEXECUTORCH_BUILD_${arg^^}=ON"
      else
        echo "Error: $arg must follow --pybind"
        exit 1
      fi
      ;;
    *)
      echo "Error: Unknown option $arg"
      exit 1
      ;;
  esac
done

# Install pytorch dependencies
#
# Note:
# When getting a new version of the executorch repo (via clone, fetch, or pull),
# you may need to re-install a new version for all pytorch dependencies to run the
# models in executorch/examples/models.
# The version in this file will be the correct version for the
# corresponsing version of the repo.
NIGHTLY_VERSION=dev20240321

TORCH_VERSION=2.4.0.${NIGHTLY_VERSION}
pip install --force-reinstall --pre torch=="${TORCH_VERSION}" -i https://download.pytorch.org/whl/nightly/cpu

TORCH_VISION_VERSION=0.19.0.${NIGHTLY_VERSION}
pip install --force-reinstall --pre torchvision=="${TORCH_VISION_VERSION}" -i https://download.pytorch.org/whl/nightly/cpu

TORCH_AUDIO_VERSION=2.2.0.${NIGHTLY_VERSION}
pip install --force-reinstall --pre torchaudio=="${TORCH_AUDIO_VERSION}" -i https://download.pytorch.org/whl/nightly/cpu

TIMM_VERSION=0.6.13
pip install --pre timm==${TIMM_VERSION}

TRANSFORMERS_VERSION=4.38.2
pip install --force-reinstall --pre transformers==${TRANSFORMERS_VERSION}

TORCHSR_VERSION=1.0.4
pip install --pre torchsr==${TORCHSR_VERSION}

# Install ExecuTorch after dependencies are installed.
EXECUTORCH_BUILD_PYBIND="$EXECUTORCH_BUILD_PYBIND" \
CMAKE_ARGS="$CMAKE_ARGS" \
CMAKE_BUILD_PARALLEL_LEVEL=9 \
pip install . --no-build-isolation -v

# Install flatc dependency
bash build/install_flatc.sh
