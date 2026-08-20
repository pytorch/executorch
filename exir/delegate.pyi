# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: delete this when pytorch pin advances

from typing import Any

import torch
from torch._ops import HigherOrderOperator

executorch_call_delegate: HigherOrderOperator

def is_lowered_module(obj: Any) -> bool: ...
def get_lowered_module_name(
    root: torch.nn.Module,
    # pyre-ignore: Undefined or invalid type [11]: Annotation `LoweredBackendModule` is not defined as a type.
    lowered_module: LOWERED_BACKEND_MODULE_TYPE,  # noqa
) -> str: ...
