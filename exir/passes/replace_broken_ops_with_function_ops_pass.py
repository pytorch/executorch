# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torch

from executorch.exir.pass_base import ExportPass


class ReplaceBrokenOpsWithFunctionalOpsPass(ExportPass):
    """
    TODO: Our backend expects pure functions. However, some operators
    are not functionalized properly. This pass intends to replace
    non-functionalized operators with their functionalized variant.

    TODO: this can be refactors into a general OpReplacementPass
    """

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op.is_view:
            namespace, op_full_name = op.name().split("::")
            split = op_full_name.split(".")
            if len(split) == 2:
                op_name, overload_name = split[0], split[1]
            elif len(split) == 1:
                # Add default overload if no overload listed
                op_name = op_full_name
                overload_name = "default"
            else:
                raise RuntimeError(
                    f"Invalid op name expected only one '.' to be present: {op_full_name}"
                )

            view_copy_op = getattr(
                getattr(getattr(torch.ops, namespace), f"{op_name}_copy"), overload_name
            )
            return super().call_operator(view_copy_op, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)
