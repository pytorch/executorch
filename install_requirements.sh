#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install required python dependencies for developing
# Dependencies are defined in .pyproject.toml
if [[ -z $PYTHON_EXECUTABLE ]];
then
  if [[ -z $CONDA_DEFAULT_ENV ]] || [[ $CONDA_DEFAULT_ENV == "base" ]] || [[ ! -x "$(command -v python)" ]];
  then
    PYTHON_EXECUTABLE=python3
  else
    PYTHON_EXECUTABLE=python
  fi
fi

if [[ "$PYTHON_EXECUTABLE" == "python" ]];
then
  PIP_EXECUTABLE=pip
else
  PIP_EXECUTABLE=pip3
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
        arg_upper="$(echo "${arg}" | tr '[:lower:]' '[:upper:]')"
        CMAKE_ARGS="$CMAKE_ARGS -DEXECUTORCH_BUILD_${arg_upper}=ON"
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

#
# Install pip packages used by code in the ExecuTorch repo.
#

# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features. The NIGHTLY_VERSION value should
# agree with the third-party/pytorch pinned submodule commit.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION=dev20240406

# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
EXIR_REQUIREMENTS=(
  torch=="2.4.0.${NIGHTLY_VERSION}"
  torchvision=="0.19.0.${NIGHTLY_VERSION}"  # For testing.
)

# pip packages needed for development.
DEVEL_REQUIREMENTS=(
  cmake  # For building binary targets.
  setuptools  # For building the pip package.
  tomli  # Imported by extract_sources.py when using python < 3.11.
  wheel  # For building the pip package archive.
  zstd  # Imported by resolve_buck.py.
)

# pip packages needed to run examples.
# TODO(dbort): Make each example publish its own requirements.txt
EXAMPLES_REQUIREMENTS=(
  timm==0.6.13
  torchaudio=="2.2.0.${NIGHTLY_VERSION}"
  torchsr==1.0.4
  transformers==4.38.2
)

# Assemble the list of requirements to actually install.
# TODO(dbort): Add options for reducing the number of requirements.
REQUIREMENTS_TO_INSTALL=(
  "${EXIR_REQUIREMENTS[@]}"
  "${DEVEL_REQUIREMENTS[@]}"
  "${EXAMPLES_REQUIREMENTS[@]}"
)

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
$PIP_EXECUTABLE install --extra-index-url "${TORCH_NIGHTLY_URL}" \
    "${REQUIREMENTS_TO_INSTALL[@]}"

#
# Install executorch pip package. This also makes `flatc` available on the path.
#

EXECUTORCH_BUILD_PYBIND="${EXECUTORCH_BUILD_PYBIND}" \
    CMAKE_ARGS="${CMAKE_ARGS}" \
    $PIP_EXECUTABLE install . --no-build-isolation -v
