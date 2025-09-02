#!/bin/bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SCRIPT_DIR=$(dirname $(readlink -fm $0))
EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))
MODEL=${1:-"cifar10"}

cd $EXECUTORCH_DIR

# Run the AoT example
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_06 -m ${MODEL}
# verify file exists
test -f ${MODEL}_nxp_delegate.pte
