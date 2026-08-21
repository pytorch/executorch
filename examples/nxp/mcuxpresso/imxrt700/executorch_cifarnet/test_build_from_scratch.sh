#!/bin/bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u
ARM_TOOLCHAIN_URL="${ARM_TOOLCHAIN_URL:-https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi.tar.xz}"

cd "$(dirname "$0")"

# Get arm gcc
echo Downloading ARM GCC toolchain
if [ ! -d arm-toolchain ]; then
    mkdir -p arm-toolchain
    pushd arm-toolchain
    wget $ARM_TOOLCHAIN_URL
    tar -xvf *.tar.xz
    rm *.tar.xz
    popd
fi
export ARMGCC_DIR=`pwd`/arm-toolchain/`ls arm-toolchain`

# Prepare model
# Side effect: the neutron SDK is installed
echo Preparing model and installing neutron SDK
./prepare_model.sh
# Check the model exists
if [ ! -f model_pte.h ]; then
    echo "Cannot create the model_pte.h!"
    exit 1;
fi

# Locate Neutron SDK
NEUTRON_LIB_DIR=`python3 -c 'exec("try:\n import eiq_neutron_sdk\n print(eiq_neutron_sdk.__path__[0])\nexcept:\n print()")'`
export NEUTRON_LIB_DIR=${NEUTRON_LIB_DIR}/target/imxrt700/rt700/cm33

# Get MCUX SDK
echo Downloading MCUXpresso SDK
if [ ! -d mcuxpresso-sdk ]; then
    pip install west
    west init -m https://github.com/nxp-mcuxpresso/mcuxsdk-manifests.git mcuxpresso-sdk
    pushd mcuxpresso-sdk
    west update_board --set board mimxrt700evk
    popd
fi
export SdkRootDirPath=`pwd`/mcuxpresso-sdk/mcuxsdk

# Build now
echo Building the example
./build_example.sh

# Test the result
if [ ! -f flash_release/executorch_cifarnet.elf ]; then
    echo "Build not successful!"
    exit 1;
else
    echo "Build successful."
    exit 0;
fi
