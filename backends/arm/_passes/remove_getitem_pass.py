# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms import remove_getitem_op
from executorch.exir.pass_base import ExportPass


class RemoveGetItemPass(ArmPass, remove_getitem_op.RemoveGetItemPass):
    _passes_required_after: Set[Type[ExportPass]] = set()
