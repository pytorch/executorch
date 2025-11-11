# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum


class DynamicMemoryPlanningMode(IntEnum):
    UPPER_BOUND = 0
    SYMBOLIC = 1
