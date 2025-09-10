# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import final, Optional, Sequence

from executorch.backends.arm.ethosu import EthosUBackend, EthosUCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir.backend.partitioner import DelegationSpec
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class EthosUPartitioner(TOSAPartitioner):
    def __init__(
        self,
        compile_spec: EthosUCompileSpec,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        # Override the delegation spec for Ethos-U
        self.delegation_spec = DelegationSpec(
            EthosUBackend.__name__, compile_spec.to_list()
        )
        self.additional_checks = additional_checks
        self.tosa_spec = compile_spec.tosa_spec
