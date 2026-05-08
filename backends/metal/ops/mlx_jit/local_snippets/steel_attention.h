/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's
 *   mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal
 *
 * Vendors the SIMD-MMA "steel" attention kernel template (Apple7-Apple8;
 * Apple9 fallback when NAX path declines, e.g. D==80 or fp32 without
 * tf32). Same redefine-instantiate-macros trick as gemv.h / sdpa_vector.h.
 *
 * Kernel template signature (steel_attention.h:60-82):
 *   attention<T, BQ, BK, BD, WM, WN, MaskType=float, AccumType=float>
 *
 * Function constants (steel_attention.h:11-16):
 *   200 align_Q
 *   201 align_K
 *   300 has_mask
 *   301 do_causal
 *   302 has_sinks
 *
 * Buffer ABI:
 *   0 Q   1 K   2 V   3 O   4 AttnParams   5 AttnMaskParams (gated)
 *   6 mask (gated)   7 sinks (gated)
 */
#pragma once

#include "mlx/backend/metal/kernels/utils.h"
// Note: steel_attention.h has no #pragma once. The .metal file below
// already #includes it — don't double-include here.

#undef  instantiate_kernel
#define instantiate_kernel(...)
#define instantiate_attn(...)
#define instantiate_attn_shapes_helper(...)
#define instantiate_attn_mask_helper(...)

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
