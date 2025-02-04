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

install_ubuntu() {
  echo "Preparing to build sccache from source"
  apt-get update
  # libssl-dev will not work as it is upgraded to libssl3 in Ubuntu-22.04.
  # Instead use lib and headers from OpenSSL1.1 installed in `install_openssl.sh``
  apt-get install -y cargo
  echo "Checking out sccache repo"
  git clone https://github.com/mozilla/sccache -b v0.8.2

  cd sccache
  echo "Building sccache"
  cargo build --release
  cp target/release/sccache /opt/cache/bin
  echo "Cleaning up"
  cd ..
  rm -rf sccache
  apt-get remove -y cargo rustc
  apt-get autoclean && apt-get clean
}

install_binary() {
  echo "Downloading sccache binary from S3 repo"
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache
  chmod +x /opt/cache/bin/sccache
}

mkdir -p /opt/cache/bin
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

install_ubuntu

function write_sccache_stub() {
  BINARY=$1
  if [ $1 == "gcc" ]; then
    # Do not call sccache recursively when dumping preprocessor argument
    # For some reason it's very important for the first cached nvcc invocation
    cat >"/opt/cache/bin/$1" <<EOF
#!/bin/sh
if [ "\$1" = "-E" ] || [ "\$2" = "-E" ]; then
  exec $(which $1) "\$@"
elif [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
EOF
  else
    cat >"/opt/cache/bin/$1" <<EOF
#!/bin/sh
if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
EOF
  fi
  chmod a+x "/opt/cache/bin/${BINARY}"
}

init_sccache() {
  # This is the remote cache bucket
  export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
  export SCCACHE_S3_KEY_PREFIX=executorch
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export RUST_LOG=sccache::server=error

  # NB: This function is adopted from PyTorch core at
  # https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/common-build.sh
  as_ci_user sccache --stop-server >/dev/null 2>&1 || true
  rm -f "${SCCACHE_ERROR_LOG}" || true

  # Clear sccache stats before using it
  as_ci_user sccache --zero-stats || true
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
init_sccache
