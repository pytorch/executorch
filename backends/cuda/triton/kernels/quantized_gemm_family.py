# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode-sized quantized GEMMs: one kernel implementation per row count M.

A quantized weight format (INT4, INT5, INT6, INT8, ...) describes its GEMM to
``QuantizedGemmFamily``: the op signature, a launcher, a fake implementation,
and one function that says why an input cannot be served. The family registers
one ``triton::<name>_m{M}`` op per supported row count ("bucket"); each is a
separate graph node, so AOTInductor compiles and tunes each bucket's kernel on
its own.

Legality lives with the format, in one place: ``supports`` (never raises) and
``validate`` (raises) both derive from the format's ``unsupported_reason``, and
every op validates its inputs before launching. Choosing a bucket for an
activation is the dispatcher's job (``quantize_op_dispatch``).

``launch_split_k_gemm`` is the launch skeleton the formats share: output and
split-K workspace allocation, the grid, and the reduce after the main kernel.
"""

from __future__ import annotations

import inspect
import typing
from typing import Callable, Optional, Sequence

import torch
import triton
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import splitk_reduce
from torch.library import triton_op, wrap_triton


def _resolved_signature(prototype) -> inspect.Signature:
    """The prototype's signature with its annotations evaluated in the
    prototype's own module, so schema inference sees real types even when that
    module uses postponed (string) annotations."""
    hints = typing.get_type_hints(prototype)
    signature = inspect.signature(prototype)
    return signature.replace(
        parameters=[
            p.replace(annotation=hints.get(p.name, p.annotation))
            for p in signature.parameters.values()
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )


class QuantizedGemmFamily:
    """The per-M ops of one quantized GEMM format.

    ``prototype`` declares the op signature, activation first, e.g.
    ``def prototype(x: Tensor, qdata: Tensor, ..., group_size: int) -> Tensor``.
    ``launch``, ``fake`` and ``unsupported_reason`` take ``(bucket, *args)`` with
    the prototype's arguments. ``unsupported_reason`` returns None when the
    bucket's kernel can serve the arguments, else a short reason; it must not
    raise for any arguments matching the prototype.
    """

    def __init__(
        self,
        name: str,
        buckets: Sequence[int],
        prototype: Callable[..., torch.Tensor],
        launch: Callable[..., torch.Tensor],
        fake: Callable[..., torch.Tensor],
        unsupported_reason: Callable[..., Optional[str]],
    ) -> None:
        if not buckets or any(b < 1 for b in buckets) or len(set(buckets)) != len(buckets):
            raise ValueError(f"buckets must be distinct positive row counts, got {buckets}")
        self.name = name
        self.buckets: tuple[int, ...] = tuple(sorted(buckets))
        self._unsupported_reason = unsupported_reason
        self._ops = {
            bucket: self._register(bucket, prototype, launch, fake)
            for bucket in self.buckets
        }

    def _register(self, bucket, prototype, launch, fake):
        op_name = f"{self.name}_m{bucket}"
        signature = _resolved_signature(prototype)

        def impl(*args):
            self.validate(bucket, *args)
            return launch(bucket, *args)

        def impl_fake(*args):
            return fake(bucket, *args)

        for fn, name in ((impl, op_name), (impl_fake, f"_{op_name}_fake")):
            fn.__name__ = fn.__qualname__ = name
            fn.__signature__ = signature
        op = triton_op(f"triton::{op_name}", mutates_args={})(impl)
        op.register_fake(impl_fake)
        return op

    def op(self, bucket: int) -> Callable[..., torch.Tensor]:
        try:
            return self._ops[bucket]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported {self.name} bucket {bucket}; expected one of {self.buckets}"
            ) from error

    def supports(self, bucket: int, *args) -> bool:
        """Whether ``op(bucket)`` can serve these arguments. Never raises."""
        return bucket in self._ops and self._unsupported_reason(bucket, *args) is None

    def validate(self, bucket: int, *args) -> None:
        """Raises RuntimeError with the reason ``op(bucket)`` cannot serve these arguments."""
        if bucket not in self._ops:
            raise RuntimeError(
                f"unsupported {self.name} bucket {bucket}; expected one of {self.buckets}"
            )
        reason = self._unsupported_reason(bucket, *args)
        if reason is not None:
            raise RuntimeError(f"{self.name}_m{bucket}: {reason}")


def launch_split_k_gemm(
    kernel,
    *,
    bucket: int,
    m,
    n: int,
    device: torch.device,
    split_k: int,
    block_m: int,
    inputs: Sequence,
    shape_args: Sequence,
    **constexprs,
) -> torch.Tensor:
    """Runs a format's main kernel and returns the BF16 [m, n] output.

    The kernel is called as ``kernel(*inputs, out, *shape_args, stride_os,
    stride_om, stride_on, SPLIT_K=split_k, **constexprs)`` on a grid of
    ``(cdiv(n, BLOCK_N), 1, split_k)``, ``BLOCK_N`` coming from its config. With
    split-K it writes FP32 partials to a ``(split_k, bucket, n)`` workspace,
    reduced deterministically afterwards. The workspace is sized by the bucket,
    not the runtime M, so its strides (constexprs) stay static under a dynamic
    M; rows at or above M are never read. Without split-K the split stride is
    unused and the kernel writes ``out`` directly.
    """
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    if split_k > 1:
        partial = torch.empty((split_k, bucket, n), dtype=torch.float32, device=device)
        stride_os = partial.stride(0)
    else:
        partial = out
        stride_os = 0

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_N"]), 1, split_k)

    wrap_triton(kernel)[grid](
        *inputs,
        partial,
        *shape_args,
        stride_os,
        partial.stride(-2),
        partial.stride(-1),
        SPLIT_K=split_k,
        **constexprs,
    )
    if split_k > 1:
        splitk_reduce(partial, out, block_m)
    return out


__all__ = ["QuantizedGemmFamily", "launch_split_k_gemm"]
