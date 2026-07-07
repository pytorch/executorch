# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA packer: assign quantized weights to model modules.

Repacks native torchao quantized tensors into the ExecuTorch-internal CUDA
layouts read by the decode kernels:

  * ``Int4Tensor`` -> ``CudaCoalescedInt4Tensor`` (bakes the scale/zero transpose
    into the coalesced [N, n_groups] layout).
  * Q5_K ``ExportableGGUFTensor`` -> ``CudaDp4aPlanarInt5Tensor`` (the genuine 5-bit
    ql/qh split bit-planes, asymmetric; the Q5_K block decode is reused from
    gguf.py, not duplicated).
  * Q6_K ``ExportableGGUFTensor`` -> ``CudaDp4aPlanarInt6Tensor`` (the genuine 6-bit
    ql/qh split bit-planes; the Q6_K block decode is reused from gguf.py, not
    duplicated).

A genuine INT8 ``IntxUnpackedToInt8Tensor`` is left unchanged for the int8 path
(Q6_K no longer arrives as an int8 tensor, so the routing is unambiguous).
The quantize_op_dispatch package (``int4_dispatch`` / ``int6_dispatch`` /
``int8_dispatch``) handles F.linear at runtime.

No CUDA is required for packing.  The backend-agnostic ``pack_model``
dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401

# ---------------------------------------------------------------------------
# Per-module packers


def pack_linear_for_cuda(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Assign a quantized weight to an ``nn.Linear`` module.

    Routes by weight type: ``Int4Tensor`` -> coalesced INT4, Q5_K
    ``ExportableGGUFTensor`` -> packed INT5, Q6_K ``ExportableGGUFTensor`` ->
    packed INT6, genuine ``IntxUnpackedToInt8Tensor`` -> int8 passthrough.
    """
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.backends.cuda.dp4a_planar_int5_tensor import (
        CudaDp4aPlanarInt5Tensor,
    )
    from executorch.backends.cuda.dp4a_planar_int6_tensor import (
        CudaDp4aPlanarInt6Tensor,
    )
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        # Convert to the ExecuTorch-internal CudaCoalescedInt4Tensor, which
        # repacks scale/zero from torchao's native [n_groups, N] layout into the
        # coalesced [N, n_groups] layout the CUDA decode kernel reads (see
        # int4_dispatch.py / int4_plain_mm.cuh). The transpose lives in
        # CudaCoalescedInt4Tensor.from_int4_tensor, so it is baked into the
        # serialized weight constant and the exported decode graph carries NO
        # per-step transpose/clone — AOTInductor (freezing=False) does not
        # constant-fold ops on parameters, so the transpose must already live in
        # the constant for the coalesced layout to pay off.
        w = CudaCoalescedInt4Tensor.from_int4_tensor(w)
    elif isinstance(w, ExportableGGUFTensor) and w.ggml_type == "q5_k":
        # GGUF Q5_K: repack the native ExportableGGUFTensor into the genuine 5-bit
        # CudaDp4aPlanarInt5Tensor (ql/qh split bit-planes, 0.625 B/elem) for the
        # W5A8 dp4a decode kernel. Q5_K is asymmetric (has dmin), so this tensor
        # carries a zero point like INT4. from_exportable_gguf reuses the shared
        # Q5_K decode (gguf.py) then bakes the bit-pack into the weight constant.
        w = CudaDp4aPlanarInt5Tensor.from_exportable_gguf(w)
    elif isinstance(w, ExportableGGUFTensor) and w.ggml_type == "q6_k":
        # GGUF Q6_K: repack the native ExportableGGUFTensor into the genuine 6-bit
        # CudaDp4aPlanarInt6Tensor (ql/qh split bit-planes, 0.75 B/elem) for the
        # W6A8 dp4a decode kernel. from_exportable_gguf reuses the shared Q6_K
        # decode (gguf.py) then bakes the bit-pack into the weight constant, once.
        w = CudaDp4aPlanarInt6Tensor.from_exportable_gguf(w)
    elif isinstance(w, IntxUnpackedToInt8Tensor):
        # Genuine INT8 weight: left unchanged for the int8 dp4a path. The
        # mixed-precision HQQ-INT4 ("sensitive") checkpoint reaches this branch
        # for its int8 tensors — edge-layer v_proj/down_proj are quantized to
        # INT8 while the rest is INT4 (see GEMMA4_31B_SENSITIVE_RECIPE in
        # quantize_and_save.py). Q6_K never reaches here (it arrives as an
        # ExportableGGUFTensor, handled above), so int4 vs int6 vs int8 routing
        # stays unambiguous.
        pass
    else:
        raise ValueError(f"Unsupported weight type: {type(w).__name__}")
    module.weight = nn.Parameter(w, requires_grad=False)


def pack_embedding_for_cuda(
    module: nn.Module, weights: dict[str, torch.Tensor]
) -> None:
    """Assign a quantized weight to an ``nn.Embedding`` (INT8 only)."""
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        raise ValueError(
            "Only 8-bit embedding quantization is supported on CUDA. "
            "INT4 does not implement the embedding op."
        )
    module.weight = nn.Parameter(w, requires_grad=False)


DEFAULT_CUDA_PACKERS: dict[type, ModulePackerFn] = {
    nn.Linear: pack_linear_for_cuda,
    nn.Embedding: pack_embedding_for_cuda,
}


# ---------------------------------------------------------------------------
# Load + pack (I/O wrapper)


def load_and_pack_for_cuda(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn] | None = None,
) -> None:
    """Load a quantized safetensors file and assign weights to the model."""
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    from .pack import pack_one

    _packers = packers or DEFAULT_CUDA_PACKERS
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())
        tensor_names = json.loads(metadata.get("tensor_names", "[]"))

        for name in tensor_names:
            parts = name.rsplit(".", 1)
            module_fqn = parts[0] if len(parts) > 1 else ""
            weight_name = parts[-1]
            prefix = (
                f"{module_fqn}._{weight_name}_" if module_fqn else f"_{weight_name}_"
            )
            partial = {}
            for key in all_keys:
                if key.startswith(prefix) or key == name:
                    partial[key] = f.get_tensor(key)
            result, _ = unflatten_tensor_state_dict(partial, metadata)
            for fqn, value in result.items():
                pack_one(model, fqn, value, _packers)

    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )
