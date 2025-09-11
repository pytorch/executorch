# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import final, List, Optional, Sequence

from executorch.backends.arm.arm_backend import (
    is_ethosu,
)  # usort: skip
from executorch.backends.arm.ethosu import EthosUBackend
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class EthosUPartitioner(TOSAPartitioner):
    def __init__(
        self,
        compile_spec: List[CompileSpec],
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        if not is_ethosu(compile_spec):
            raise RuntimeError("compile spec is not targeting Ethos-U")

        # Override the delegation spec for Ethos-U
        self.delegation_spec = DelegationSpec(EthosUBackend.__name__, compile_spec)
        self.additional_checks = additional_checks
