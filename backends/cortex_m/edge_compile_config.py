# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import EdgeCompileConfig

# Ops that must survive to_edge for the Cortex-M passes to lower them directly.
# Left to decompose, silu becomes sigmoid plus an elementwise mul and hardswish
# becomes a clamp plus a mul, which costs an extra kernel and, for hardswish,
# quantizes the gate on the producer's grid instead of using the exact int8 LUT.
_PRESERVE_OPS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid_.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.silu.default,
)


def cortex_m_edge_compile_config() -> EdgeCompileConfig:
    """The to_edge configuration the Cortex-M backend requires.

    Shared by the AOT compiler and the test harness so the two cannot drift: an
    entry present in only one of them means the tests exercise a lowering users
    never get, or the reverse.
    """
    return EdgeCompileConfig(
        preserve_ops=list(_PRESERVE_OPS),
        _check_ir_validity=False,
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    )
