/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's
 *   mlx/backend/metal/kernels/quantized.metal
 *
 * Vendors MLX's affine-quantized kernel templates (qmv_fast / qmv /
 * qmv_quad / qmm_t / qmm_n / qvm / etc.) for use by
 * AffineQuantizedLinearOp via per-shape JIT. The .metal file
 * unconditionally expands hundreds of AOT instantiate_quantized*(...)
 * calls; this wrapper redefines them to no-ops so only the templates
 * survive in the preprocessed source.
 *
 * Pattern matches sdpa_vector.h / steel_attention.h. Note that
 * quantized.h, quantized_utils.h, etc. lack `#pragma once` — DO NOT
 * explicitly #include them here. The .metal file includes them.
 *
 * The downstream awk filter in make_mlx_jit_snippet.sh additionally
 * strips any leftover `#define instantiate_quantized*` lines and
 * top-level `instantiate_quantized*(...);` calls so the generated
 * .cpp's R-string preamble is free of AOT [[host_name(...)]] entries.
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

#include "mlx/backend/metal/kernels/quantized.metal"
