# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-free weight representation conversion.

Two ``convert`` helpers map a raw torchao quantized weight to its portable
``Exportable*`` form. They are ``(fqn, tensor) -> tensor`` so the same function
drives ``iter_checkpoint`` (streaming from disk) and ``assign_state_dict`` (an
in-memory quantized state dict), both in ``extension/llm/export/load.py``:

  * ``to_exportable`` wraps a torchao ``Int4Tensor`` as
    ``ExportableInt4Tensor`` -- so it exports to the portable
    ``dequantize_int4_tensor`` op instead of torchao's hardware-specific ``mslk``
    kernels -- and passes every other tensor through unchanged.
  * ``to_default`` (the default ``convert``) additionally repacks a 4-bit
    ``IntxUnpackedToInt8Tensor`` (stored one value per int8 byte) into the
    nibble-packed ``ExportableInt4Tensor`` to halve its footprint.
  * ``identity`` yields the raw tensor/subclass untouched.

These only convert a weight's *representation*; assigning it into a model
(``assign_one`` / ``assign_state_dict``) and backend-specific layout conversion
(CUDA coalesced int4, MLX gather buffers) live elsewhere.

:func:`fuse_along_output` is the multi-weight companion: it concatenates several
weights (e.g. ``gate_proj|up_proj`` or ``q|k|v``) along the output-channel dim,
subclass-aware, so fusion happens once on the portable form before backend
packing. It handles the same tensor types as :func:`maybe_cast`.
"""

from collections.abc import Sequence
from typing import Callable, Optional

import torch

# A weight conversion step: (fqn, tensor) -> tensor. Runs after key remapping,
# so it may key off the model FQN; the built-ins below ignore the name.
Convert = Callable[[str, torch.Tensor], torch.Tensor]


def maybe_cast(value: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Cast a weight to ``dtype`` only when it is safe and meaningful.

    * ``dtype`` is None / already matches / non-floating -> returned unchanged.
    * a plain (unquantized) tensor -> cast normally.
    * a quantized subclass whose ``.to`` preserves the payload
      (``ExportableGGUFTensor``, ``ExportableInt4Tensor``,
      ``IntxUnpackedToInt8Tensor`` -- only ``scale``/``zero_point`` and the
      dequantized output dtype change) -> cast via that ``.to``.
    * any other subclass -> returned unchanged, so the cast can never silently
      dequantize it (via an ``aten._to_copy`` override) or no-op it (torchao's
      own ``Int4Tensor.to`` ignores dtype).

    Shared by ``iter_checkpoint`` (which casts the *converted* weight, so a raw
    ``Int4Tensor`` is wrapped as an ``ExportableInt4Tensor`` first and can then be
    re-stamped, e.g. to fp16 for MLX) and ``quantize_stream`` (which casts each
    weight after applying the recipe), so both cast identically.
    """
    if dtype is None or value.dtype == dtype or not value.dtype.is_floating_point:
        return value
    if type(value) is torch.Tensor:
        return value.to(dtype)

    from executorch.extension.llm.export import (
        ExportableGGUFTensor,
        ExportableInt4Tensor,
        ExportableMXTensor,
        ExportableNVFP4Tensor,
    )
    from torchao.quantization import IntxUnpackedToInt8Tensor

    castable = (
        ExportableGGUFTensor,
        ExportableInt4Tensor,
        ExportableMXTensor,
        ExportableNVFP4Tensor,
        IntxUnpackedToInt8Tensor,
    )
    return value.to(dtype) if isinstance(value, castable) else value


def _is_quantized(value: torch.Tensor) -> bool:
    """Check if a tensor is a torchao quantized subclass."""
    from torchao.utils import TorchAOBaseTensor

    return isinstance(value, TorchAOBaseTensor)


def fuse_along_output(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate weights along the output-channel (``N``) dim, subclass-aware.

    Fuses e.g. ``gate_proj|up_proj`` or ``q|k|v`` into one weight *before*
    backend packing, so both the MLX and CUDA packers consume a single fused
    weight (no backend-specific concat needed). ``N`` is dim 0 of the logical
    ``(N, K)`` weight. Handles the same types as :func:`maybe_cast`:

    * a plain (unquantized) tensor -> ``torch.cat(dim=0)``.
    * ``ExportableInt4Tensor`` -> ``qdata`` on dim 0, ``scale``/``zero_point`` on
      dim 1 (they are stored transposed as ``(K // group_size, N)``).
    * ``IntxUnpackedToInt8Tensor`` / ``ExportableGGUFTensor`` -> every packed
      field on dim 0 (all are ``N``-major).

    Exact: ``scale``/``zero_point`` are per-(group, output-channel) and each
    packed row is an independent group / GGUF super-block, so concatenating
    output channels leaves every channel's metadata untouched. All inputs must
    be the same type and share their quant params + inner (``K``) shape;
    ``torch.cat`` enforces the shapes and a mismatched attribute raises
    ``ValueError``. A single tensor is returned unchanged; an empty input or an
    unsupported subclass raises.
    """
    tensors = list(tensors)
    if not tensors:
        raise ValueError("fuse_along_output requires at least one tensor")
    first = tensors[0]
    if len(tensors) == 1:
        return first

    # Fusability: all inputs must be the same kind. Plain weights must all be
    # plain; quantized weights must be the exact same subclass (with matching
    # attributes, checked in _fuse_subclass). torch.cat then enforces the inner
    # (K) shape.
    if not _is_quantized(first):
        if any(_is_quantized(t) for t in tensors):
            raise TypeError(
                "fuse_along_output cannot mix plain and quantized tensors; got "
                f"{[type(t).__name__ for t in tensors]}"
            )
        return torch.cat(tensors, dim=0)
    if any(type(t) is not type(first) for t in tensors):
        raise TypeError(
            "fuse_along_output requires all tensors to be the same type; got "
            f"{[type(t).__name__ for t in tensors]}"
        )

    from executorch.extension.llm.export import (
        ExportableGGUFTensor,
        ExportableInt4Tensor,
        ExportableMXTensor,
        ExportableNVFP4Tensor,
    )
    from torchao.quantization import IntxUnpackedToInt8Tensor

    # Per-data-field concat axis (the output-channel dim) by subclass layout. An
    # axis of ``None`` means the field is not per-output-channel (e.g. NVFP4's
    # scalar ``per_tensor_scale``): it must match across inputs and is kept as-is.
    axis_by_field = {
        ExportableInt4Tensor: {"qdata": 0, "scale": 1, "zero_point": 1},
        IntxUnpackedToInt8Tensor: {"qdata": 0, "scale": 0, "zero_point": 0},
        ExportableGGUFTensor: {"raw": 0},
        # NVFP4/MX store qdata (N, ...) and scale (N, K//group) N-major.
        ExportableNVFP4Tensor: {"qdata": 0, "scale": 0, "per_tensor_scale": None},
        ExportableMXTensor: {"qdata": 0, "scale": 0},
    }.get(type(first))
    if axis_by_field is None:
        raise TypeError(
            f"fuse_along_output cannot fuse tensors of type {type(first).__name__}"
        )
    return _fuse_subclass(tensors, axis_by_field)


def _fuse_subclass(
    tensors: list[torch.Tensor], axis_by_field: dict[str, int]
) -> torch.Tensor:
    """Rebuild a TorchAO subclass from its data fields concatenated per axis.

    Relies on the ``(*tensor_data_names, *tensor_attribute_names)`` constructor
    convention shared by these subclasses; attributes must match across inputs.
    """
    first = tensors[0]
    attrs = [getattr(first, name) for name in first.tensor_attribute_names]
    for t in tensors[1:]:
        for name, ref in zip(first.tensor_attribute_names, attrs):
            if getattr(t, name) != ref:
                raise ValueError(
                    "fuse_along_output: cannot fuse tensors with different "
                    f"'{name}' ({getattr(t, name)!r} != {ref!r})"
                )
    fused_data = []
    for name in first.tensor_data_names:
        axis = axis_by_field[name]
        if axis is None:
            # Not per-output-channel (e.g. a scalar per-tensor scale): must
            # match across all inputs; keep the first.
            ref = getattr(first, name)
            for t in tensors[1:]:
                if not torch.equal(getattr(t, name), ref):
                    raise ValueError(
                        "fuse_along_output: cannot fuse tensors with different "
                        f"'{name}'"
                    )
            fused_data.append(ref)
        else:
            fused_data.append(torch.cat([getattr(t, name) for t in tensors], dim=axis))
    return type(first)(*fused_data, *attrs)


def _to_exportable_int4(w: torch.Tensor, *, repack_intx: bool) -> torch.Tensor:
    """Wrap 4-bit torchao weights as ``ExportableInt4Tensor``.

    Always converts a torchao ``Int4Tensor``. When ``repack_intx`` is set, also
    repacks a 4-bit ``IntxUnpackedToInt8Tensor`` into the nibble-packed int4
    representation (raising if it carries activation quantization, which int4
    export can't represent). Any other tensor is returned unchanged.
    """
    from executorch.extension.llm.export import ExportableInt4Tensor
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    if isinstance(w, Int4Tensor):
        return ExportableInt4Tensor.from_int4_tensor(w)
    if (
        repack_intx
        and isinstance(w, IntxUnpackedToInt8Tensor)
        and w.target_dtype == torch.int4
    ):
        return ExportableInt4Tensor.from_intx_unpacked_to_int8_tensor(w)
    return w


def to_exportable(fqn: str, w: torch.Tensor) -> torch.Tensor:
    """Default convert: ``Int4Tensor -> ExportableInt4Tensor``; others unchanged.

    ``Int4Tensor`` is wrapped so it exports to the portable
    ``dequantize_int4_tensor -> linear/embedding`` op (torchao's own
    ``Int4Tensor`` would otherwise lower to hardware-specific ``mslk`` kernels).
    All other subclasses -- including 4-bit ``IntxUnpackedToInt8Tensor`` -- already
    export to portable ``dequantize_affine`` and are passed through.
    """
    return _to_exportable_int4(w, repack_intx=False)


def to_default(fqn: str, w: torch.Tensor) -> torch.Tensor:
    """Like :func:`to_exportable`, but also repacks a 4-bit
    ``IntxUnpackedToInt8Tensor`` into the nibble-packed ``ExportableInt4Tensor``
    (halving its footprint vs the int8 container).

    Opt-in (e.g. the MLX gemma4 export), since intx already exports portably via
    ``dequantize_affine`` and int4 is purely a memory optimization here.
    """
    return _to_exportable_int4(w, repack_intx=True)


def identity(fqn: str, w: torch.Tensor) -> torch.Tensor:
    """No-op convert: yield the raw tensor/subclass unchanged."""
    return w
