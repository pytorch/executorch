# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convenience specs for the coreai-torch composite op library.

The specs themselves are ``coreai_torch.ExternalizeSpec``. Only the target
classes are named here; each one's composite attributes are read off the class
rather than transcribed, so this cannot drift from the installed SDK.
"""

from typing import Any, List, Tuple

import torch

# Carried by generate_composite_decl's own version parameter, so it is not a
# composite attribute. Private names are implementation detail.
_NOT_COMPOSITE_ATTRS = frozenset({"version"})

# nn.Module's own bookkeeping, computed so a torch version change cannot leave
# a hand-written list stale.
_MODULE_INTERNALS = frozenset(vars(torch.nn.Module()))

# Composite op name to the coreai_torch.composite_ops class it externalizes.
_TARGETS = {
    "rms_norm": "RMSNormImpl",
    "rope": "RoPE",
    "scaled_dot_product_attention": "SDPA",
    "gather_mm": "GatherMM",
    "gated_delta_update": "GatedDeltaUpdate",
}


def target_class(composite_op_name: str) -> type:
    """The composite_ops class externalized as ``composite_op_name``."""
    import coreai_torch.composite_ops as composite_ops

    return getattr(composite_ops, _TARGETS[composite_op_name])


def composite_attrs(module: torch.nn.Module) -> Tuple[str, ...]:
    """Composite attribute names implied by a module instance.

    Every library composite declares exactly its public non-tensor attributes
    minus ``version``, so reading them off the instance keeps the spec honest
    against the installed SDK.
    """
    return tuple(
        sorted(
            name
            for name, value in vars(module).items()
            if name not in _MODULE_INTERNALS
            and name not in _NOT_COMPOSITE_ATTRS
            and not name.startswith("_")
            and not isinstance(value, (torch.Tensor, torch.nn.Module))
        )
    )


def spec_for(composite_op_name: str) -> Any:
    """Build the ``ExternalizeSpec`` for one library composite."""
    from coreai_torch import ExternalizeSpec

    cls = target_class(composite_op_name)
    return ExternalizeSpec(
        target_class=cls,
        composite_op_name=composite_op_name,
        composite_attrs=list(composite_attrs(cls())),
    )


def default_specs() -> List[Any]:
    """Specs for every composite in ``coreai_torch.composite_ops``."""
    return [spec_for(name) for name in _TARGETS]
