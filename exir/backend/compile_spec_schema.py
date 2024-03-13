# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

"""
Please refer to executorch/schema/program.fbs for source of truth.
"""


@dataclass
class CompileSpec:
    key: str  # like max_value
    value: bytes  # like 4 or other types
