# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.argmax.default` / `aten.argmin.default` modules for the op-test framework.

Last-dim arg-reduction -> int64 index. The kernel writes an int32 index (the AOT
downcasts the int64 output), which `copy_outputs` widens to the int64 program
output. `golden_dtype="float32"` so the golden's fp32 argmax matches the kernel's
fp32 argmax exactly (avoids an fp32-vs-fp64 tie-break discriminator flip).
"""

import torch


class ArgmaxModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=-1)


class ArgminModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmin(x, dim=-1)


def argmax_tie_gen(shape):
    # Repeated max in the last dim (idx 1 and 3); the FIRST wins (index 1) —
    # discriminates the strict-`>` tie-break from a `>=` (last-wins) bug.
    assert shape[-1] >= 4, "tie generator writes idx 1 and 3; last dim must be >= 4"
    base = torch.arange(shape[-1], dtype=torch.float32) * 0.01
    t = base.expand(shape).contiguous()
    t[..., 1] = 10.0
    t[..., 3] = 10.0
    return t


def argmin_tie_gen(shape):
    # Repeated min in the last dim (idx 1 and 3); the FIRST wins (index 1).
    assert shape[-1] >= 4, "tie generator writes idx 1 and 3; last dim must be >= 4"
    base = 5.0 + torch.arange(shape[-1], dtype=torch.float32) * 0.01
    t = base.expand(shape).contiguous()
    t[..., 1] = -10.0
    t[..., 3] = -10.0
    return t
