# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve submodule boundaries as Core AI composite ops through ExecuTorch.

Externalization itself belongs to coreai-torch, which splits it into two calls
with the caller's export in between: patch the matching submodules, export,
sub-export each one. :func:`externalize_modules` runs all three, since
ExecuTorch has nothing to do in between and a backend never sees the
``nn.Module``, so the boundary must be captured before ``torch.export``.
``TorchConverter`` turns the result into ``noinline`` composite graphs; the
rest of this package carries the prepared submodules through to ``preprocess``.

    from executorch.backends.apple.coreai.externalize import (
        default_specs,
        externalize_modules,
    )

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

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from coreai_torch import (
    _patch_model_for_externalization,
    _subexport_and_restore,
    ExternalizeSpec,
)

# Private upstream; see docs/externalize_spec.md for the request to publish
# NAMESPACE and the externalized-module type. The restore helpers are needed
# only because _subexport_and_restore cannot unpatch a model whose export
# raised.
from coreai_torch._utils import _EXTERNALIZE_NAMESPACE as NAMESPACE
from coreai_torch.externalize import (
    _ExternalizedExportedProgram as ExternalizedModule,
    _find_marked_submodules,
    _restore_externalized,
)

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
from torch.export.exported_program import ExportedProgram


def externalize_modules(
    model: torch.nn.Module,
    targets: Sequence[Union[type, ExternalizeSpec]],
    *,
    export_fn: Callable[[torch.nn.Module], ExportedProgram],
) -> Tuple[ExportedProgram, List[ExternalizedModule]]:
    """Patch, export, and sub-export in one call.

    coreai-torch keeps the patch and the sub-export apart so a caller can
    quantize in between. ExecuTorch has no such step and needs both results
    together for ``CoreAIPartitioner(externalized_modules=...)``, so they run
    as one call here.

    The model is left unpatched even when ``export_fn`` raises, which
    ``_subexport_and_restore`` cannot handle itself: it owns the restore but
    never runs if the export it consumes failed.

    Args:
        model: Model to externalize. Not mutated once this returns.
        targets: ``ExternalizeSpec`` objects, or bare classes.
        export_fn: Exports the patched model. Must use a decomposition table
            that preserves the composite ops expected to survive.

    Returns:
        The whole-model program containing the custom op call sites, and one
        prepared submodule per call site.
    """
    _patch_model_for_externalization(model, targets)
    try:
        exported_program = export_fn(model)
    except Exception:
        _restore_externalized(_find_marked_submodules(model))
        raise
    return exported_program, _subexport_and_restore(model, exported_program)


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
    "externalize_modules",
    "externalized_op_name",
    "is_externalize_target",
    "is_supported_target",
    "lookup",
    "register",
    "spec_for",
]
