#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

python -m pip install build 'setuptools>=77' 'cmake>=3.26,<4' ninja auditwheel patchelf
python -m unittest discover -s "${NATIVE_WHEEL_ROOT}/.ci/scripts/tests" \
    -p test_native_wheel_version.py
mkdir -p "${NATIVE_WHEEL_DEPS}"
conda create -y -p "${NATIVE_WHEEL_DEPS}/shaderc" -c conda-forge shaderc zlib

curl --fail --location --retry 3 https://libzip.org/download/libzip-1.11.4.tar.xz \
    --output "${NATIVE_WHEEL_DEPS}/libzip.tar.xz"
tar -xJf "${NATIVE_WHEEL_DEPS}/libzip.tar.xz" -C "${NATIVE_WHEEL_DEPS}"
cmake -S "${NATIVE_WHEEL_DEPS}/libzip-1.11.4" -B "${NATIVE_WHEEL_DEPS}/libzip-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${NATIVE_WHEEL_DEPS}/libzip" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TOOLS=OFF -DBUILD_REGRESS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOC=OFF \
    -DENABLE_BZIP2=OFF -DENABLE_LZMA=OFF -DENABLE_ZSTD=OFF \
    -DENABLE_OPENSSL=OFF -DENABLE_GNUTLS=OFF -DENABLE_MBEDTLS=OFF
cmake --build "${NATIVE_WHEEL_DEPS}/libzip-build" --target install --parallel

cmake -S "${NATIVE_WHEEL_ROOT}/third-party/flatbuffers" \
    -B "${NATIVE_WHEEL_DEPS}/flatbuffers" \
    -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF
cmake --build "${NATIVE_WHEEL_DEPS}/flatbuffers" --target flatc --parallel
