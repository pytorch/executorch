# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass

from executorch.backends.arm._passes.convert_int64_const_ops_to_int32 import (
    ConvertInt64ConstOpsToInt32Pass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes import remove_graph_asserts_pass


class RemoveGraphAssertsPass(remove_graph_asserts_pass.RemoveGraphAssertsPass, ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {ConvertInt64ConstOpsToInt32Pass}
