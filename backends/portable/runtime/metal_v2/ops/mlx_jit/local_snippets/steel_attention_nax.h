/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's
 *   mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal
 *
 * Vendors the NAX "steel" attention kernel (Apple9+ cooperative-tensor
 * matmul). Same pattern as steel_attention.h.
 *
 * Kernel template (steel_attention_nax.h):
 *   attention_nax<T, BQ, BK, BD, WM, WN, MaskType=float, AccumType=float>
 *
 * Function constants:
 *   200 align_Q
 *   201 align_K
 *   300 has_mask
 *   301 do_causal
 *   302 has_sinks
 *
 * Buffer ABI: same as steel_attention.h.
 *
 * NAX requires the Apple-private MetalPerformancePrimitives header to be
 * resolvable on the runtime metal compiler's include path. xcrun's metal
 * compiler ships it, so the runtime newLibraryWithSource path picks it up
 * the same way our existing NAX matmul snippet (steel_gemm_fused_nax)
 * already does.
 */
#pragma once

#include "mlx/backend/metal/kernels/utils.h"
// Note: steel_attention_nax.h has no #pragma once. The .metal file below
// already #includes it — don't double-include here.

#undef  instantiate_kernel
#define instantiate_kernel(...)
#define instantiate_attn(...)
#define instantiate_attn_shapes_helper(...)
#define instantiate_attn_mask_helper(...)

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal"
