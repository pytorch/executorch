# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule


def functional_aten_op(target) -> object:
    """Return the functional ATen overload for an edge ``*_copy`` op, or ``None``.

    ExecuTorch's edge dialect functionalizes view ops into ``*_copy`` variants
    (``permute_copy``, ``view_copy``, ``slice_copy``, ...).  ``coreai-torch``
    only registers the non-copy forms (``permute``, ``view``, ``slice``), so we
    map ``aten::permute_copy.default`` -> ``torch.ops.aten.permute.default``.

    Returns ``None`` for anything that is not an edge ``*_copy`` op or whose
    functional counterpart cannot be resolved.
    """
    if not isinstance(target, EdgeOpOverload):
        return None
    name = target._op.__name__  # e.g. "permute_copy.default"
    op_name, _, overload = name.partition(".")
    if not op_name.endswith("_copy"):
        return None
    base = op_name[: -len("_copy")]
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        return None
    return getattr(packet, overload or "default", None)


class ReplaceCopyOpsWithFunctionalPass:
    """Preprocess-only pass: retarget edge ``*_copy`` ops to their functional
    ATen forms so ``coreai-torch`` can lower them.

    Only rewrites when ``coreai-torch`` actually supports the functional form;
    unsupported ops are left untouched so conversion still raises an informative
    error rather than silently dropping the op.  This is safe to run only inside
    the backend's ``preprocess`` because view/copy semantics are equivalent in
    Core AI's value-based IR. It must not be applied to the shared ExecuTorch
    edge graph, which relies on the functionalized ``*_copy`` forms.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        from executorch.backends.apple.coreai.partition.partitioner import (
            is_coreai_supported_target,
        )

        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            func_op = functional_aten_op(node.target)
            if func_op is not None and is_coreai_supported_target(func_op):
                node.target = func_op
                modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
