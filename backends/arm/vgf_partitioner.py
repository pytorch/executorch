# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import final, List, Optional, Sequence

from executorch.backends.arm.arm_backend import (
    is_vgf,
)  # usort: skip
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf_backend import VgfBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class VgfPartitioner(TOSAPartitioner):
    def __init__(
        self,
        compile_spec: List[CompileSpec],
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        if not is_vgf(compile_spec):
            raise RuntimeError("compile spec is not targeting Vgf")

        # Override the delegation spec for Vgf
        self.delegation_spec = DelegationSpec(VgfBackend.__name__, compile_spec)
        self.additional_checks = additional_checks
