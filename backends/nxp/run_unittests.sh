#!/bin/bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SCRIPT_DIR=$(dirname $(readlink -fm $0))
EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))

cd $EXECUTORCH_DIR

# '-c /dev/null' is used to ignore root level pytest.ini.
pytest -c /dev/null backends/nxp/tests/

python -m unittest discover -s backends/nxp/tests/ -v
