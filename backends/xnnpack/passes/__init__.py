# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from executorch.backends.xnnpack.passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack.passes.conv1d_unsqueeze_pass import Conv1dUnsqueezePass
from executorch.backends.xnnpack.passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.xnnpack.passes.prelu_reshape_pass import PReLUReshapePass
from executorch.backends.xnnpack.passes.remove_getitem_op import RemoveGetItemPass
from executorch.backends.xnnpack.passes.tag_implicit_q_dq_pass import TagImplicitQDqPass
from executorch.backends.xnnpack.passes.xnnpack_pass import XNNPACKPass

from executorch.exir.pass_base import ExportPass

from executorch.exir.passes.const_prop_pass import ConstPropPass
from torch._export.pass_base import PassType

from torch.export import ExportedProgram


class XNNPACKPassManager:
    def __init__(
        self, exported_program: ExportedProgram, passes: Optional[List[PassType]] = None
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
                ConvertToLinearPass,
                ConstPropPass,
                FuseBatchNormWithConvPass,
                RemoveGetItemPass,
                Conv1dUnsqueezePass,
                PReLUReshapePass,
                ChannelsLastTaggedReshapePass,
                TagImplicitQDqPass,
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
            ep = ep._transform(transform_pass)
        return ep
