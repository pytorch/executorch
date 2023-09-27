#!/bin/bash
# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -e

#
# Setup toolchain
#
BASEDIR=`realpath $(dirname "$0")`
echo "building using build.sh in $BASEDIR"

GCCPATH=${BASEDIR}/arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi/bin/
echo $GCCPATH
if test -d "${GCCPATH}"; then
	echo Using exising compiler ${GCCPATH}
else
	pushd ${BASEDIR}/
	./toolchain.sh
	popd
fi
export PATH=${PATH}:${GCCPATH}

echo building with `arm-none-eabi-gcc -v 2>&1 | grep "^gcc"`


#
# Prepare and run clean build
#
rm -rf buck-out/ build/lib/ cmake-out/
rm -rf cmake-corstone
mkdir cmake-corstone
cd cmake-corstone

#cmake -DBUCK2=buck2 ..

#cmake --toolchain backends/arm/cmake/arm-none-eabi-gcc.cmake ..
cmake -DFLATC_EXECUTABLE=flatc \
	  -DEXECUTORCH_BUILD_HOST_TARGETS=OFF \
	  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
	  -DCMAKE_SYSTEM_PROCESSOR=cortex-m55+nodsp+nofp \
	  -DETHOSU_TARGET_NPU_CONFIG=ethos-u55-128 \
	  --toolchain backends/arm/cmake/arm-none-eabi-gcc.cmake \
	  ..
# -DCMAKE_TOOLCHAIN_FILE=backends/arm/cmake/arm-none-eabi-gcc.cmake \

cd ..
cmake --build cmake-corstone -j1 --target ethos_u ethosu_core_driver
