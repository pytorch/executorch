# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared config.json loader for the ToyModel training example scripts."""

import json
import os


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
