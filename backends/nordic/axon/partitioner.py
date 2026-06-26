# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU partitioner for ExecuTorch.

Delegates supported operations to the AXON NPU backend.
Reuses TOSAPartitioner for TOSA-compatible operation checking,
with additional AXON hardware constraint checks that reject nodes
exceeding tensor size limits, input count, or filter dimensions.
"""

from __future__ import annotations

from typing import final

from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir.backend.partitioner import DelegationSpec

from .backend import AxonBackend
from .compile_spec import AxonCompileSpec


@final
class AxonPartitioner(TOSAPartitioner):
    """Partitioner that delegates supported operations to AXON NPU.

    Inherits from TOSAPartitioner to reuse TOSA-based operator support
    checks. Adds AXON-specific hardware constraint checks that reject
    nodes exceeding:

    - Max tensor dimensions (1024 height/width/channels)
    - Max input count per node (2)
    - Max FC input/output (2048)
    - Max Conv2D filter size (16x16) and stride (31)

    Nodes that pass TOSA checks but fail AXON constraints fall back
    to CPU execution via ExecuTorch's portable kernels.
    """

    def __init__(
        self,
        compile_spec: AxonCompileSpec,
        additional_checks=None,
    ):
        self.compile_spec = compile_spec
        self.delegation_spec = DelegationSpec(
            AxonBackend.__name__,
            compile_spec.to_compile_specs(),
        )

        # AXON hardware constraint checks can be added via additional_checks.
        # By default, TOSA-level checks handle partitioning. For stricter
        # enforcement of AXON-specific limits (tensor dim 1024, FC 2048,
        # Conv filter 16, max 2 inputs), pass get_axon_constraint_checks():
        #
        #   from executorch.backends.nordic.operator_support.axon_constraints import (
        #       get_axon_constraint_checks,
        #   )
        #   partitioner = AxonPartitioner(spec, additional_checks=get_axon_constraint_checks())
        self.additional_checks = additional_checks or []

        # Use TOSA INT profile for operator support checking
        from executorch.backends.arm.tosa.specification import TosaSpecification
        self.tosa_spec = TosaSpecification.create_from_string(compile_spec.tosa_spec)
