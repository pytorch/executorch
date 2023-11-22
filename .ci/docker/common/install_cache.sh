#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NB: This script is adopted from PyTorch core repo at
# https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cache.sh
set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_binary() {
  echo "Downloading sccache binary from S3 repo"
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache
  chmod +x /opt/cache/bin/sccache
}

mkdir -p /opt/cache/bin
mkdir -p /opt/cache/lib
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

# NB: Install the pre-built binary from S3 as building from source
# https://github.com/pytorch/sccache has started failing mysteriously
# in which sccache server couldn't start with the following error:
#   sccache: error: Invalid argument (os error 22)
install_binary

function write_sccache_stub() {
  BINARY=$1
  printf "#!/bin/sh\nif [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then\n  exec sccache %s \"\$@\"\nelse\n  exec %s \"\$@\"\nfi" "$(which "${BINARY}")" "$(which "${BINARY}")" > "/opt/cache/bin/${BINARY}"
  chmod a+x "/opt/cache/bin/${BINARY}"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
init_sccache
