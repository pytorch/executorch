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
"""

from __future__ import annotations

import inspect
import typing
from typing import Callable, Optional, Sequence

import torch
from torch.library import triton_op


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
        if (
            not buckets
            or any(b < 1 for b in buckets)
            or len(set(buckets)) != len(buckets)
        ):
            raise ValueError(
                f"buckets must be distinct positive row counts, got {buckets}"
            )
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


__all__ = ["QuantizedGemmFamily"]
