# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-internal INT4 tensor for the CUDA W4A8 dp4a decode kernel.

``CudaCoalescedInt4Tensor`` is an ExecuTorch-internal tensor subclass. It is
**NOT** torchao's ``Int4Tensor`` and is intentionally not a subclass of it, so
torchao's ``Int4Tensor`` F.linear handlers never match it via the method
resolution order. The CUDA decode/prefill dispatch (``int4_dispatch.py``) is
selected by *type* — it is registered on this class only — so stock
``Int4Tensor`` weights keep falling back to torchao's default (mslk/tinygemm)
path.

Layout difference from torchao ``Int4Tensor``:
    qdata      : packed int4 weight (N, K/2), nibble-packed (same as Int4Tensor)
    scale      : (N, n_groups) — the *coalesced* layout, transposed from
                 torchao's documented (n_groups, N)
    zero_point : (N, n_groups) — coalesced, transposed from (n_groups, N)

The coalesced [N, n_groups] layout is exactly what the W4A8 dp4a matvec kernel
(``executorch_cuda::int4_plain_mm`` / ``int4_plain_mm.cuh``) reads row-for-row
with qdata, so the exported decode graph carries no per-step transpose. The
transpose is owned by :meth:`from_int4_tensor` so it is baked into the
serialized weight constant once at pack time.
"""

from typing import List, Optional

import torch
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "CudaCoalescedInt4Tensor",
]


class CudaCoalescedInt4Tensor(TorchAOBaseTensor):
    """INT4 weight with scale/zero_point in the coalesced [N, n_groups] layout.

    ExecuTorch-internal; see the module docstring. Mirrors torchao
    ``Int4Tensor``'s data/attribute layout (so the common tensor utilities and
    serialization work) but owns the [n_groups, N] -> [N, n_groups] transpose
    of scale/zero_point via :meth:`from_int4_tensor`.
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]
    optional_tensor_attribute_names = ["activation_dtype"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.activation_dtype = (
            activation_dtype if activation_dtype is not None else torch.bfloat16
        )
        self.act_pre_scale = act_pre_scale

    def _quantization_type(self):
        s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}, activation_dtype={self.activation_dtype}"
        if self.act_pre_scale is not None:
            s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_int4_tensor(cls, t: Int4Tensor) -> "CudaCoalescedInt4Tensor":
        """Build a coalesced tensor from a torchao ``Int4Tensor``.

        Owns the transpose: torchao stores scale/zero_point as (n_groups, N);
        the CUDA decode kernel reads (N, n_groups). The ``.t().contiguous()``
        here is baked into the serialized weight constant so the exported
        decode graph has no per-step transpose/clone.
        """
        return cls(
            t.qdata,
            t.scale.t().contiguous(),
            t.zero_point.t().contiguous(),
            t.block_size,
            t.shape,
            t.act_pre_scale,
            t.activation_dtype,
        )


# Allow a model with CudaCoalescedInt4Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao Int4Tensor).
torch.serialization.add_safe_globals([CudaCoalescedInt4Tensor])
