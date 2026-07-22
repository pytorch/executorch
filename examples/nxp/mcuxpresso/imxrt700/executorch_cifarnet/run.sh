#!/bin/bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [ -z ${ARMGCC_DIR+x} ]; then
    echo "ARMGCC_DIR needs to be set in the environment!"
    exit 1;
fi

if [ -z ${SdkRootDirPath+x} ]; then
    echo "SdkRootDirPath needs to be set in the environment!"
    exit 1;
fi

mkdir cmake-out
cd cmake-out
cmake -DSdkRootDirPath=${SdkRootDirPath} \
      -DCMAKE_TOOLCHAIN_FILE=${SdkRootDirPath}/tools/cmake_toolchain_files/armgcc.cmake \
      -DCMAKE_BUILD_TYPE=flash_release \
      -G "Unix Makefiles" \
      -DINTTYPES_FORMAT:STRING=C99 \
      ..
make -j 6 executorch_cifarnet.elf