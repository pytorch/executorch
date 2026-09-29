#!/bin/bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SCRIPT_DIR=$(dirname $(readlink -fm $0))
EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))

cd $EXECUTORCH_DIR

# Cap pytest-xdist's workers to the container's CPU quota. Applies to
# `-n logical` as well, despite the variable's name.
source .ci/scripts/pytest-parallelism.sh

# '-c /dev/null' is used to ignore root level pytest.ini.
pytest -c /dev/null -n "logical" backends/nxp/tests/

python -m unittest discover -s backends/nxp/tests/ -v
