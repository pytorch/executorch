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

if [ ! -f model_pte.h ]; then
    echo "Cannot find model_pte.h!"
    exit 1;    
fi

if [ -z ${NEUTRON_LIB_DIR+x} ]; then
    echo "NEUTRON_LIB_DIR needs to be set in the environment!"
    exit 1;
fi

if [ ! -f ${NEUTRON_LIB_DIR}/libNeutronDriver.a ]; then
    echo "Neutron driver not found in ${NEUTRON_LIB_DIR}!"
    exit 1;    
fi

if [ ! -f ${NEUTRON_LIB_DIR}/libNeutronFirmware.a ]; then
    echo "Neutron firmware not found in ${NEUTRON_LIB_DIR}!"
    exit 1;    
fi

rm -rf cmake-out && mkdir -p cmake-out

cmake -DSdkRootDirPath=${SdkRootDirPath} \
      -DCMAKE_TOOLCHAIN_FILE=${SdkRootDirPath}/cmake/toolchain/armgcc.cmake \
      -DMODEL_DIR=$(pwd) \
      -DNEUTRON_LIB_DIR=${NEUTRON_LIB_DIR} \
      -DCMAKE_BUILD_TYPE=flash_release \
      -G "Unix Makefiles" \
      -B cmake-out \
      "$(dirname "$0")"

make -C cmake-out -j$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) + 1 )) executorch_cifarnet.elf
