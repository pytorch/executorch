# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve submodule boundaries as Core AI composite ops through ExecuTorch.

Externalization itself belongs to coreai-torch. ``externalize_modules`` marks
the matching submodules, exports the model, and prepares each submodule;
``TorchConverter`` turns the result into ``noinline`` composite graphs. This
package only supplies what ExecuTorch adds: the boundary has to be captured
before ``torch.export`` because a backend never sees the ``nn.Module``, and the
prepared submodules have to reach ``preprocess``.

    from coreai_torch import externalize_modules
    from executorch.backends.apple.coreai.externalize import default_specs

    ep, externalized = externalize_modules(
        model, default_specs(), export_fn=my_export_fn
    )
    lowered = to_edge_transform_and_lower(
        ep,
        partitioner=[CoreAIPartitioner(externalized_modules=externalized)],
    )

Specs are ``coreai_torch.ExternalizeSpec``, so the same list also drives
``TorchConverter.add_pytorch_module(externalize_modules=...)``.
"""

from typing import Any, Optional, Sequence

from coreai_torch import ExternalizedModule, ExternalizeSpec

# The namespace coreai-torch marks externalized submodules with. Private
# upstream; see docs/externalize_spec.md for the request to publish it.
from coreai_torch._utils import _EXTERNALIZE_NAMESPACE as NAMESPACE

from executorch.backends.apple.coreai.compiler.fx_targets import (
    target_name,
    target_namespace,
)
from executorch.backends.apple.coreai.externalize.registry import (
    clear,
    lookup,
    register,
)
from executorch.backends.apple.coreai.externalize.specs import default_specs, spec_for


def is_externalize_target(target: Any) -> bool:
    """Whether an fx target is an externalized submodule's custom op.

    Works on both edge and ATen targets; the partitioner sees the former and
    the converter the latter.
    """
    return target_namespace(target) == NAMESPACE


def externalized_op_name(target: Any) -> str:
    """The op name coreai-torch derived from the submodule's path."""
    return (target_name(target) or "").split(".")[0]


def is_supported_target(
    target: Any, externalized: Optional[Sequence[ExternalizedModule]]
) -> bool:
    """Whether the partitioner should claim this externalized op.

    An op with no prepared submodule behind it must stay out of the delegate.
    Claiming it would emit a program whose delegate cannot lower it.
    """
    if not is_externalize_target(target):
        return False
    if not externalized:
        return False
    return externalized_op_name(target) in {e.op_name for e in externalized}


__all__ = [
    "ExternalizeSpec",
    "ExternalizedModule",
    "NAMESPACE",
    "clear",
    "default_specs",
    "externalized_op_name",
    "is_externalize_target",
    "is_supported_target",
    "lookup",
    "register",
    "spec_for",
]
