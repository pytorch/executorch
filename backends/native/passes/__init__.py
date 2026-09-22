# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph transformation passes for the native backend.

These are intended to be passed to ``to_edge_transform_and_lower`` via its
``transform_passes`` argument, so they run on the edge-dialect graph before
partitioning/delegation:

    to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

reinplace rewrites functional ops into their in-place forms (e.g. relu -> relu_),
which are not in the core-ATen opset, so callers must disable IR validity
checking via ``EdgeCompileConfig(_check_ir_validity=False)`` (the same config the
MLX backend requires).

Each pass lives in its own module; this package aggregates them and exposes the
default pass list.
"""

from typing import List, Union

from executorch.backends.native.passes.fuse_rope import FuseRoPEPass
from executorch.backends.native.passes.reinplace import (
    backend_inplace_aten_variants,
    BACKEND_INPLACE_OPS,
    NativeReinplacePass,
)
from executorch.backends.native.passes.replace_copy_with_alias import (
    ReplaceCopyWithAliasPass,
)

from executorch.backends.transforms.collapse_view_copy import CollapseViewCopyPass
from executorch.backends.transforms.fuse_gqa_with_sdpa import FuseGQAWithSDPAPass
from executorch.backends.transforms.fuse_rms_norm import FuseRMSNormPass
from executorch.backends.transforms.normalize_sdpa_input_rank import (
    NormalizeSDPAInputRankPass,
)

from executorch.exir.pass_base import ExportedProgramPassBase, ExportPass
from executorch.exir.passes.cse_pass import CSEPass

__all__ = [
    "backend_inplace_aten_variants",
    "BACKEND_INPLACE_OPS",
    "CollapseViewCopyPass",
    "FuseGQAWithSDPAPass",
    "FuseRMSNormPass",
    "FuseRoPEPass",
    "get_default_passes",
    "NativeReinplacePass",
    "NormalizeSDPAInputRankPass",
    "ReplaceCopyWithAliasPass",
]


def get_default_passes() -> List[Union[ExportPass, ExportedProgramPassBase]]:
    """Passes enabled by default for the native backend.

    GQA fusion runs before SDPA rank normalization, then view_copy collapsing
    and CSE settle the graph. Reinplace runs last, rewriting functional ops into
    their in-place edge forms.
    """
    return [
        FuseRoPEPass(),
        FuseGQAWithSDPAPass(),
        NormalizeSDPAInputRankPass(),
        FuseRMSNormPass(fold_dtype_casts=True, allow_lossy_weight_casts=True),
        CollapseViewCopyPass(),
        CSEPass(),
        NativeReinplacePass(),
    ]
