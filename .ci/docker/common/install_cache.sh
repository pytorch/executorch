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
  echo "Downloading sccache binary from GitHub"
  curl --retry 3 "https://github.com/mozilla/sccache/releases/download/v0.7.3/sccache-v0.7.3-x86_64-unknown-linux-musl.tar.gz" -o sccache.tar.gz
  tar xfz sccache.tar.gz

  cp sccache/sccache /opt/cache/bin/sccache
  chmod a+x /opt/cache/bin/sccache
}

mkdir -p /opt/cache/bin
mkdir -p /opt/cache/lib
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

# Setup compiler cache
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

# NB: Install the pre-built binary from S3 as building from source
# https://github.com/pytorch/sccache has started failing mysteriously
# in which sccache server couldn't start with the following error:
#   sccache: error: Invalid argument (os error 22)
install_binary

function write_sccache_stub() {
  # Unset LD_PRELOAD for ps because of asan + ps issues
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90589
  printf "#!/bin/sh\nif [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then\n  exec sccache $(which $1) \"\$@\"\nelse\n  exec $(which $1) \"\$@\"\nfi" > "/opt/cache/bin/$1"
  chmod a+x "/opt/cache/bin/$1"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
init_sccache
