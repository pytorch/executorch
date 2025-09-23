#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_ubuntu() {
  apt-get update
  apt-get install -y zstd python3-certifi
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  
  # Create a cache directory for buck2
  CACHE_DIR="/tmp/buck2_cache"
  mkdir -p "${CACHE_DIR}"

  # Run resolve_buck.py to get the buck2 binary
  BUCK2_PATH=$(python "${REPO_ROOT}/tools/cmake/resolve_buck.py" --cache_dir "${CACHE_DIR}")

  # Move buck2 to /usr/bin/
  mv "${BUCK2_PATH}" /usr/bin/buck2
  chmod +x /usr/bin/buck2

  # Cleanup
  rm -rf "${CACHE_DIR}"
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
