# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import final, Optional, Sequence

from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfBackend, VgfCompileSpec
from executorch.exir.backend.partitioner import DelegationSpec
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class VgfPartitioner(TOSAPartitioner):
    def __init__(
        self,
        compile_spec: VgfCompileSpec,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        # Override the delegation spec for Vgf
        self.delegation_spec = DelegationSpec(
            VgfBackend.__name__, compile_spec.to_list()
        )
        self.additional_checks = additional_checks
        self.tosa_spec = compile_spec.tosa_spec
