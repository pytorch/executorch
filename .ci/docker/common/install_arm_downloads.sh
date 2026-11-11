#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Cache the developer.arm.com downloads of examples/arm/setup.sh, so CI jobs
# copy them from ARM_SETUP_DOWNLOAD_CACHE instead of fetching them each run.
# Jobs download whatever is missing, so a failed download only costs the cache.

set -exu

ARCH="$(uname -m)"
OS="$(uname -s)"
ARM_SETUP_CURL_PROGRESS_ARGS=(--no-progress-meter)
# Fill the cache rather than read from it
cache_dir="${ARM_SETUP_DOWNLOAD_CACHE}"
unset ARM_SETUP_DOWNLOAD_CACHE
source ./arm-setup/fvp_utils.sh
source ./arm-setup/toolchain_utils.sh

cache_download() {
    local url=$1
    local md5=$2
    local output="${cache_dir}/$(basename "${url}")"
    download_with_retry "cache" "${url}" "${output}" "${md5}" || rm -f "${output}"
}

mkdir -p "${cache_dir}"
cache_download "${corstone300_url}" "${corstone300_md5_checksum}"
cache_download "${corstone320_url}" "${corstone320_md5_checksum}"
gcc_select_toolchain
cache_download "${toolchain_url}" "${toolchain_md5_checksum}"
