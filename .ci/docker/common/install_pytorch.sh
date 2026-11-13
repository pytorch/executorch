#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# The compiler stubs run through sccache, which fails hard rather than
# compiling uncached when it cannot reach its S3 bucket. Use the credentials
# docker-builds mounts, or fall back to a local cache so a build without them
# still works.
SCCACHE_CREDENTIALS=/run/secrets/aws-credentials
if [[ -s "${SCCACHE_CREDENTIALS}" ]]; then
  export AWS_SHARED_CREDENTIALS_FILE="${SCCACHE_CREDENTIALS}"
else
  echo "No sccache credentials; caching compiler output locally" >&2
  unset SCCACHE_BUCKET SCCACHE_S3_KEY_PREFIX
  export SCCACHE_DIR=/tmp/sccache
fi

configure_pytorch_compiler() {
  local cxx_version
  cxx_version=$(/usr/bin/c++ --version | head -n1)
  if [[ "${cxx_version}" == *clang* ]]; then
    local clang_major
    clang_major=$(/usr/bin/c++ -dumpversion | cut -d. -f1)
    if [[ "${clang_major}" =~ ^[0-9]+$ && "${clang_major}" -lt 16 ]] &&
      command -v gcc >/dev/null &&
      command -v g++ >/dev/null; then
      export CC=gcc
      export CXX=g++
    fi
  fi
}

configure_pytorch_compiler
export TORCH_VERSION
export PYTHON_EXECUTABLE="/opt/conda/envs/py_${PYTHON_VERSION}/bin/python"
conda_run bash "$(dirname "${BASH_SOURCE[0]}")/install_pinned_pytorch.sh"

if [[ "${TORCHCI_REPORTS:-}" == "1" ]]; then
  (cd / && conda_run python -c '
import os
import torch
from torch.testing._internal.torchci import environment, plugin, report

if torch.version.git_version != os.environ["TORCH_VERSION"]:
    raise RuntimeError("Installed torch differs from the image source pin")
if report.SCHEMA_VERSION != "1.0":
    raise RuntimeError("The image requires torchci report schema 1.0")
environment.capture()
')
fi
