#!/bin/bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SCRIPT_DIR=$(dirname $(readlink -fm $0))
EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))
MODEL=${1:-"cifar10"}

cd ${EXECUTORCH_DIR}

echo "** Build nxp_executor_runner"
if [ ! -d ${SCRIPT_DIR}/executor_runner/build ] ; then
  mkdir ${SCRIPT_DIR}/executor_runner/build
fi
rm -rf ${SCRIPT_DIR}/executor_runner/build/*

pushd ${SCRIPT_DIR}/executor_runner/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 nxp_executor_runner
popd

echo "** Export cifar10 model to executorch"
# Run the AoT example
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_12 -m "cifar10"
test -f cifar10_nxp_delegate.pte

echo "** Generate test dataset"
python -m examples.nxp.experimental.cifar_net.cifar_net --store-test-data

echo "** Run nxp_executor_runner"
${SCRIPT_DIR}/executor_runner/build/nxp_executor_runner \
    --firmware `make -C ${SCRIPT_DIR}/executor_runner/build locate_neutron_firmware | grep "NeutronFirmware.elf" ` \
    --nsys `which nsys` \
    --nsys_config ${SCRIPT_DIR}/executor_runner/neutron-imxrt700.ini \
    --model cifar10_nxp_delegate.pte \
    --dataset ./cifar10_test_data \
    --output ./cifar10_test_output
