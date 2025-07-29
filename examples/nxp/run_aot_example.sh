#!/bin/bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SCRIPT_DIR=$(dirname $(readlink -fm $0))
EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))

cd $EXECUTORCH_DIR

# Run the AoT example
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_03 -m cifar10
# verify file exists
test -f cifar10_nxp_delegate.pte
