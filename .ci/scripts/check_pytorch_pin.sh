#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script checks if the PyTorch commit hash in pytorch.txt matches
# the commit hash for the NIGHTLY_VERSION specified in torch_pin.py.
#
# It calls the Python script check_pytorch_pin.py which uses functions
# from update_pytorch_pin.py to fetch the expected commit hash and
# compare it with the current pin.

set -eu

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python check script
python3 "$SCRIPT_DIR/check_pytorch_pin.py"
