#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

current_file=os.path.basename(__file__)
print(f"\033[31m[error] build/{current_file} has moved to:\033[0m scripts/{current_file}")
sys.exit(1)
