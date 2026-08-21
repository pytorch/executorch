# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import EdgeCompileConfig

# Ops that must survive to_edge for the Cortex-M passes to lower them directly.
# Omitting one does not degrade gracefully. The activations decompose into a
# multiply whose qparams are gone by then, which fails AtenToCortexMPass; linear
# decomposes into addmm and silently stays on portable float kernels.
_PRESERVE_OPS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid_.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.silu.default,
)


def cortex_m_edge_compile_config(
    use_explicit_layout: bool = False,
) -> EdgeCompileConfig:
    """The to_edge configuration the Cortex-M backend requires.

    Shared by the AOT compiler and the test harness so the two cannot drift: an
    entry present in only one of them means the tests exercise a lowering users
    never get, or the reverse.

    Edge-dialect validation is off because the backend's quantized graphs do not
    pass it: enabling it fails the model tests with mismatched-dtype
    SpecViolationErrors. That also makes a _core_aten_ops_exception_list pointless
    here, since the verifier it feeds never runs. max_pool2d would need an entry if
    validation is ever turned on.
    """
    return EdgeCompileConfig(
        preserve_ops=list(_PRESERVE_OPS),
        _check_ir_validity=False,
        _skip_dim_order=use_explicit_layout,
    )
