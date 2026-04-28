# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cuda.triton.kernels.fused_moe import (
    fused_moe,
    fused_moe_batched,
    fused_moe_batched_gemm,
    moe_align_block_size,
)
from executorch.backends.cuda.triton.kernels.sdpa import sdpa, sdpa_decode_splitk
from executorch.backends.cuda.triton.kernels.topk import topk

from executorch.backends.cuda.triton.kernels.int4_matmul import (
    dequant_w4_to_bf16,
    int4_matvec,
)

__all__ = [
    "dequant_w4_to_bf16",
    "fused_moe",
    "fused_moe_batched",
    "fused_moe_batched_gemm",
    "moe_align_block_size",
    "sdpa",
    "sdpa_decode_splitk",
    "topk",
]

try:
    from executorch.backends.cuda.triton.kernels.chunk_gated_delta_rule import (  # noqa: F401
        chunk_gated_delta_rule,
    )

    __all__.append("chunk_gated_delta_rule")
except ImportError:
    pass

try:
    from executorch.backends.cuda.triton.kernels.tq4_sdpa import tq4_sdpa  # noqa: F401

    __all__.append("tq4_sdpa")
except ImportError:
    pass

try:
    from executorch.backends.cuda.triton.kernels.fused_deltanet_decode import (  # noqa: F401
        fused_deltanet_decode,
    )

    __all__.append("fused_deltanet_decode")
except ImportError:
    pass
