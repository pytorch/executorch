#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Before doing anything, cd to the directory containing this script.
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null || /bin/true

# Find the names of the python tools to use.
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

# Returns 0 if the current python version is compatible with the version range
# in pyprojects.toml, or returns 1 if it is not compatible. If the check logic
# itself fails, prints a warning and returns 0.
python_is_compatible() {
  # Scrape the version range from pyproject.toml, which should be
  # in the current directory.
  local version_specifier
  version_specifier="$(
    grep "^requires-python" pyproject.toml \
      | head -1 \
      | sed -e 's/[^"]*"//;s/".*//'
  )"
  if [[ -z ${version_specifier} ]]; then
    echo "WARNING: Skipping python version check: version range not found" >& 2
    return 0
  fi

  # Install the packaging module if necessary.
  if ! python -c 'import packaging' 2> /dev/null ; then
    ${PIP_EXECUTABLE} install packaging
  fi

  # Compare the current python version to the range in version_specifier. Exits
  # with status 1 if the version is not compatible, or with status 0 if the
  # version is compatible or the logic itself fails.
${PYTHON_EXECUTABLE} <<EOF
import sys
try:
  import packaging.version
  import packaging.specifiers
  import platform

  python_version = packaging.version.parse(platform.python_version())
  version_range = packaging.specifiers.SpecifierSet("${version_specifier}")
  if python_version not in version_range:
    print(
        "ERROR: ExecuTorch does not support python version "
        + f"{python_version}: must satisfy \"${version_specifier}\"",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
  print(f"WARNING: Skipping python version check: {e}", file=sys.stderr)
  sys.exit(0)
EOF

  return $?
}

# Fail fast if the wheel build will fail because the current python version
# isn't supported. But don't fail if the check logic itself has problems: the
# wheel build will do a final check before proceeding.
if ! python_is_compatible; then
  exit 1
fi

# Parse options.
EXECUTORCH_BUILD_PYBIND=OFF

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
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION=dev20240716

# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
EXIR_REQUIREMENTS=(
  torch=="2.5.0.${NIGHTLY_VERSION}"
  torchvision=="0.20.0.${NIGHTLY_VERSION}"  # For testing.
)

# pip packages needed for development.
DEVEL_REQUIREMENTS=(
  cmake  # For building binary targets.
  "pip>=23" # For building the pip package.
  pyyaml  # Imported by the kernel codegen tools.
  "setuptools>=63"  # For building the pip package.
  tomli  # Imported by extract_sources.py when using python < 3.11.
  wheel  # For building the pip package archive.
  zstd  # Imported by resolve_buck.py.
)

# pip packages needed to run examples.
# TODO(dbort): Make each example publish its own requirements.txt
EXAMPLES_REQUIREMENTS=(
  timm==1.0.7
  torchaudio=="2.4.0.${NIGHTLY_VERSION}"
  torchsr==1.0.4
  transformers==4.42.4
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
# The --extra-index-url may be necessary if pyproject.toml has a dependency on a
# pre-release or nightly version of a torch package.
#
CMAKE_ARGS="$CMAKE_ARGS -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"
EXECUTORCH_BUILD_PYBIND="${EXECUTORCH_BUILD_PYBIND}" \
    CMAKE_ARGS="${CMAKE_ARGS}" \
    CMAKE_BUILD_ARGS="${CMAKE_BUILD_ARGS}" \
    $PIP_EXECUTABLE install . --no-build-isolation -v \
        --extra-index-url "${TORCH_URL}"
