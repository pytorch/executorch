/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's
 *   mlx/backend/metal/kernels/quantized_nax.metal
 *
 * Vendors the NAX (Apple9+) variants of MLX's affine-quantized kernels.
 * Same pattern as quantized.h.
 */
#pragma once

#include "mlx/backend/metal/kernels/utils.h"

#undef  instantiate_kernel
#define instantiate_kernel(...)
#define instantiate_quantized(...)
#define instantiate_quantized_batched(...)
#define instantiate_quantized_aligned(...)
#define instantiate_quantized_aligned_batched(...)
#define instantiate_quantized_quad(...)
#define instantiate_quantized_split_k(...)
#define instantiate_quantized_batched_wrap(...)
#define instantiate_quantized_all_batched(...)
#define instantiate_quantized_all_single(...)
#define instantiate_quantized_all_aligned(...)
#define instantiate_quantized_all_quad(...)
#define instantiate_quantized_all_splitk(...)
#define instantiate_quantized_splitk_qmm(...)
#define instantiate_quantized_all_splitk_qmm(...)
#define instantiate_quantized_all_rhs(...)
#define instantiate_quantized_funcs(...)
#define instantiate_quantized_types(...)
#define instantiate_quantized_groups(...)
#define instantiate_quantized_all(...)

#include "mlx/backend/metal/kernels/quantized_nax.metal"
