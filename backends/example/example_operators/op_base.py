# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class OpBase:
    pattern: Tuple[Callable]
    annotate_handle: Callable
    permuate_memory_format: bool = False
