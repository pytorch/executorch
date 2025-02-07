#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# This script is run after building ExecuTorch binaries

# Rename pip-out directory, to avoid using shared libraries in pip-out during
# smoke test.
mv pip-out BACKUP-pip-out
