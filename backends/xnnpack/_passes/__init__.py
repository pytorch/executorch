# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Type

from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform

from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass

from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack._passes.conv1d_unsqueeze_pass import (
    Conv1dUnsqueezePass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.convert_to_sdpa import ConvertToSDPAPass
from executorch.backends.xnnpack._passes.convert_to_upsample_bilinear2d import (
    ConvertToUpsampleBilinear2d,
)
from executorch.backends.xnnpack._passes.decompose_cat import DecomposeConcatenate
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack._passes.fuse_batch_norm import FuseBatchNormPass
from executorch.backends.xnnpack._passes.prelu_reshape_pass import PReLUReshapePass
from executorch.backends.xnnpack._passes.remove_redundant_copy_pass import (
    RemoveRedundantCopyPass,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass

from executorch.exir.pass_base import ExportPass

from executorch.exir.passes.const_prop_pass import ConstPropPass
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass

from executorch.exir.program._program import _transform
from torch._export.pass_base import PassType

from torch.export import ExportedProgram

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class XNNPACKPassManager:
    def __init__(
        self,
        exported_program: ExportedProgram,
        passes: Optional[List[Type[PassType]]] = None,
    ) -> None:
        """
        A helper class to run multiple XNNPACK passes on a program
        If passes list is empty, all passes in XNNPACK will be run.
        Else only run passes in the list will be run.
        """
        self._exported_program = exported_program

        if not passes:
            # All the XNNPACK passes
            self.passes = [
                # TODO - remove this pass once we have a better support for dim_order ops lowering
                DimOrderOpsRevertPass,
                ConvertToUpsampleBilinear2d,
                ConvertToLinearPass,
                ConvertToSDPAPass,
                ConstPropPass,
                FuseBatchNormPass,
                FuseActivationPass,
                DecomposeConcatenate,
                RemoveGetItemPass,
                Conv1dUnsqueezePass,
                PReLUReshapePass,
                ChannelsLastTaggedReshapePass,
                RemoveRedundantCopyPass,
                RemoveCloneOpsTransform,
            ]
        else:
            self.passes = passes

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    def transform(self) -> ExportedProgram:
        """
        Returns a transformed ExportedProgram
        """
        ep = self.exported_program
        for pass_ in self.passes:
            if issubclass(pass_, XNNPACKPass):
                transform_pass = pass_(ep)
            elif issubclass(pass_, ExportPass):
                transform_pass = pass_()
            else:
                raise RuntimeError(
                    f"Expecting ExportPass or ExportPass(), but got pass: {pass_} with type: {type(pass_)}"
                )
            ep = _transform(ep, transform_pass)
            logger.debug(f"Running {pass_.__name__} pass")
            logger.debug(f"Transformed program: {ep}")
        return ep
