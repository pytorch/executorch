# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Type

from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import _transform

from torch._export.pass_base import PassType
from torch.export import ExportedProgram


class NeutronPassManager:
    def __init__(
        self,
        exported_program: ExportedProgram,
        passes: Optional[List[Type[PassType]]] = None,
    ) -> None:
        """
        A helper class to run multiple passes on a program
        """
        self._exported_program = exported_program

        if not passes:
            self.passes = []
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
            if issubclass(pass_, ExportPass):
                transform_pass = pass_()
            else:
                raise RuntimeError(
                    f"Expecting ExportPass or ExportPass(), but got pass: {pass_} with type: {type(pass_)}"
                )
            ep = _transform(ep, transform_pass)
        return ep
