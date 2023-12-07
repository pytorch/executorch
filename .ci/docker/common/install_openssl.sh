#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NB: This script is needed for sccache and is adopted from PyTorch core repo at
# https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_openssl.sh
set -ex

OPENSSL=openssl-1.1.1k

wget -q -O "${OPENSSL}.tar.gz" "https://ossci-linux.s3.amazonaws.com/${OPENSSL}.tar.gz"
tar xf "${OPENSSL}.tar.gz"

pushd "${OPENSSL}" || true
./config --prefix=/opt/openssl -d "-Wl,--enable-new-dtags,-rpath,$(LIBRPATH)"
# NOTE: openssl install errors out when built with the -j option
make -j6; make install_sw
# Link the ssl libraries to the /usr/lib folder.
ln -s /opt/openssl/lib/lib* /usr/lib
popd || true

rm -rf "${OPENSSL}"
