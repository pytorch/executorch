#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

current_file=$(basename "$0")
echo -e "\033[31m[error] $0 has moved to:\033[0m scripts/${current_file}"
exit 1
