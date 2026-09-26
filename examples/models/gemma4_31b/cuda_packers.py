# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA weight conversion pass: rewrite portable quantized weights to CUDA layouts.

``convert_quantized_tensors_for_cuda`` is a terminal pass over a loaded model:
it walks every ``nn.Linear`` / ``nn.Embedding`` and repacks its quantized weight
into the ExecuTorch-internal CUDA layouts read by the decode kernels. It runs
after ``load_checkpoint`` (and any fusion), consuming the canonical portable
forms that loading produced (``ExportableInt4Tensor``, ``ExportableGGUFTensor``,
int8 ``IntxUnpackedToInt8Tensor``).

Linear layouts:
  * ``ExportableInt4Tensor`` -> ``CudaCoalescedInt4Tensor``
    (bakes the scale/zero transpose into the coalesced [N, n_groups] layout). A
    Q4_K ``ExportableGGUFTensor`` is decoded to an ``ExportableInt4Tensor`` first,
    reaching the same layout from the GGUF blob.
  * Q5_K ``ExportableGGUFTensor`` -> ``CudaDp4aPlanarInt5Tensor`` (5-bit ql/qh
    split bit-planes, asymmetric).
  * Q6_K ``ExportableGGUFTensor`` -> ``CudaDp4aPlanarInt6Tensor`` (6-bit ql/qh
    split bit-planes).
  * genuine INT8 ``IntxUnpackedToInt8Tensor`` -> left unchanged (int8 dp4a path).

Embedding layouts:
  * GGUF ``ExportableGGUFTensor`` (e.g. the tied token embedding) -> gatherable
    int8 ``IntxUnpackedToInt8Tensor``.
  * genuine int8 ``IntxUnpackedToInt8Tensor`` -> unchanged.
  * int4 (``Exportable``/``Int4Tensor``) -> raises (no int4 embedding op).

No CUDA is required for packing. ``load_checkpoint`` lives in
``extension/llm/export/load.py``; the model-free conversion helpers in
``extension/llm/export/quant/convert.py``.
"""

import torch
import torch.nn as nn

from executorch.extension.llm.export.quant.convert import _is_quantized

# ---------------------------------------------------------------------------
# Per-module weight conversion


def pack_linear_for_cuda(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Convert an ``nn.Linear``'s quantized weight to its CUDA layout, in place.

    Routes by weight type: ``ExportableInt4Tensor`` (or Q4_K
    ``ExportableGGUFTensor``) -> coalesced INT4, Q5_K ``ExportableGGUFTensor`` ->
    packed INT5, Q6_K ``ExportableGGUFTensor`` -> packed INT6, genuine
    ``IntxUnpackedToInt8Tensor`` -> int8 passthrough.
    """
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.backends.cuda.dp4a_planar_int5_tensor import (
        CudaDp4aPlanarInt5Tensor,
    )
    from executorch.backends.cuda.dp4a_planar_int6_tensor import (
        CudaDp4aPlanarInt6Tensor,
    )
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from executorch.extension.llm.export.int4 import ExportableInt4Tensor
    from torchao.quantization import IntxUnpackedToInt8Tensor

    w = weights["weight"]
    if isinstance(w, ExportableInt4Tensor):
        # Canonical portable int4 (what load_checkpoint produces by default).
        # CudaCoalescedInt4Tensor reuses the nibble-packed qdata untouched and
        # re-encodes scale/zero into the coalesced [N, n_groups] layout the CUDA
        # decode kernel reads (int4_dispatch.py / int4_plain_mm.cuh). The
        # transpose is baked into the serialized constant, so the exported decode
        # graph carries no per-step transpose/clone.
        w = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(w)
    elif isinstance(w, ExportableGGUFTensor) and w.ggml_type == "q4_k":
        # GGUF Q4_K: decode to an ExportableInt4Tensor (bf16 — fp16 is not a CUDA
        # target), then coalesce as above. Same end state as the safetensors int4
        # path; the GGUF blob is just the other source.
        w = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(
            w.to_exportable_int4_tensor(torch.bfloat16)
        )
    elif isinstance(w, ExportableGGUFTensor) and w.ggml_type == "q5_k":
        # GGUF Q5_K: genuine 5-bit CudaDp4aPlanarInt5Tensor (ql/qh split
        # bit-planes, 0.625 B/elem) for the W5A8 dp4a decode kernel. Asymmetric
        # (has dmin), so it carries a zero point like INT4. from_exportable_gguf
        # reuses the shared Q5_K decode (gguf.py) and bakes the bit-pack in.
        w = CudaDp4aPlanarInt5Tensor.from_exportable_gguf(w)
    elif isinstance(w, ExportableGGUFTensor) and w.ggml_type == "q6_k":
        # GGUF Q6_K: genuine 6-bit CudaDp4aPlanarInt6Tensor (ql/qh split
        # bit-planes, 0.75 B/elem) for the W6A8 dp4a decode kernel.
        # from_exportable_gguf reuses the shared Q6_K decode (gguf.py), once.
        w = CudaDp4aPlanarInt6Tensor.from_exportable_gguf(w)
    elif isinstance(w, IntxUnpackedToInt8Tensor):
        # Genuine INT8 weight: left unchanged for the int8 dp4a path. The
        # mixed-precision HQQ-INT4 ("sensitive") checkpoint reaches this branch
        # for its int8 tensors — edge-layer v_proj/down_proj are quantized to
        # INT8 while the rest is INT4 (see GEMMA4_31B_SENSITIVE_RECIPE in
        # quantize_and_save.py). Q6_K never reaches here (it arrives as an
        # ExportableGGUFTensor, handled above), so int4/int6/int8 routing stays
        # unambiguous.
        pass
    else:
        raise ValueError(f"Unsupported weight type: {type(w).__name__}")
    module.weight = nn.Parameter(w, requires_grad=False)


def pack_embedding_for_cuda(
    module: nn.Module, weights: dict[str, torch.Tensor]
) -> None:
    """Convert an ``nn.Embedding``'s quantized weight to gatherable int8, in place.

    A GGUF ``ExportableGGUFTensor`` (Q4_K/Q6_K, e.g. the tied token embedding)
    is decoded to a gatherable int8 ``IntxUnpackedToInt8Tensor`` -- the packed
    int4/int6 matmul layouts can't gather, and the lossless int8 decode halves
    the footprint vs bf16. A genuine int8 ``IntxUnpackedToInt8Tensor`` (the
    safetensors int8 embedding) is assigned unchanged. INT4 is unsupported (no
    int4 embedding op).
    """
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from executorch.extension.llm.export.int4 import ExportableInt4Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, ExportableGGUFTensor):
        w = w.to_intx_unpacked_to_int8_tensor(scale_dtype=torch.bfloat16)
    elif isinstance(w, (Int4Tensor, ExportableInt4Tensor)):
        raise ValueError(
            "Only 8-bit embedding quantization is supported on CUDA. "
            "INT4 does not implement the embedding op."
        )
    module.weight = nn.Parameter(w, requires_grad=False)


# ---------------------------------------------------------------------------
# Model pass


def convert_quantized_tensors_for_cuda(model: nn.Module) -> None:
    """Rewrite every quantized ``nn.Linear`` / ``nn.Embedding`` weight in ``model``
    into its CUDA layout, in place.

    Terminal pass: run after ``load_checkpoint`` and any fusion. Walks the model,
    dispatches by module type (the tied-embedding split: ``lm_head`` ->
    ``CudaCoalescedInt4Tensor``, ``embed_tokens`` -> gatherable int8), and mutates
    each module's ``weight``. Plain (unquantized) weights are left untouched.
    """
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is None or not _is_quantized(weight):
            continue
        if isinstance(module, nn.Linear):
            pack_linear_for_cuda(module, {"weight": weight})
        elif isinstance(module, nn.Embedding):
            pack_embedding_for_cuda(module, {"weight": weight})
        else:
            raise ValueError(
                f"Cannot convert quantized weight on unsupported module type "
                f"{type(module).__name__}: only nn.Linear and nn.Embedding are "
                f"handled by the CUDA pass."
            )
