# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import traceback
from typing import Optional

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata


class ArmPass(ExportPass):
    """Base class for Arm passes"""

    def __init__(self, exported_program: Optional[torch.export.ExportedProgram] = None):
        super(ArmPass, self).__init__()
        self.exported_program = exported_program

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if not updated:
            return super().call_operator(op, args, kwargs, meta)

        # if updated we should update metadata
        new_meta = {}
        keys = meta.data.keys()
        for key in keys:
            new_meta[key] = meta[key]
        old_stack_trace = new_meta.get("stack_trace", "")
        new_meta["stack_trace"] = f"{old_stack_trace}\n{traceback.format_stack()[-2]}"
        return super().call_operator(op, args, kwargs, NodeMetadata(new_meta))
