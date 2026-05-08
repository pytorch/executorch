/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's
 *   mlx/backend/metal/kernels/scaled_dot_product_attention.metal
 *
 * Mirrors the gemv.h wrapper pattern. The MLX file:
 *   - defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`
 *     kernel TEMPLATES (via #include "mlx/.../sdpa_vector.h")
 *   - then unconditionally expands ~24 AOT instantiate_kernel(...) calls via
 *     `instantiate_sdpa_vector_heads(float|float16_t|bfloat16_t)`
 *
 * We want only the templates — the runtime metal compiler will JIT each
 * (T, D, V) instantiation we actually need, parameterized by the
 * function-constant slots 20-26 from sdpa_vector.h.
 *
 * Trick (same as gemv.h):
 *   1. Pull in MLX's shared headers (utils.h here transitively brings in
 *      defines.h's `instantiate_kernel` macro).
 *   2. Undef + redefine `instantiate_kernel` (and the local helper macros
 *      defined inside scaled_dot_product_attention.metal itself) to no-ops
 *      BEFORE including the .metal file.
 *   3. Include the .metal file. Its includes are header-guarded → no
 *      redefinition. Its body's instantiation macros expand to nothing.
 *
 * The downstream awk filter in make_mlx_jit_snippet.sh additionally
 * strips any leftover `#define instantiate_*` lines and top-level
 * `instantiate_*(...)` calls so the generated .cpp's R-string preamble
 * is free of AOT [[host_name(...)]] entries.
 */
#pragma once

// 1. Pull in MLX's shared utils.h (which transitively #includes defines.h
//    where `instantiate_kernel` is defined). DO NOT explicitly #include
//    sdpa_vector.h here — it has no header guard / #pragma once and
//    scaled_dot_product_attention.metal already includes it.
#include "mlx/backend/metal/kernels/utils.h"

// 2. Suppress AOT instantiation macros — both the generic `instantiate_kernel`
//    and the SDPA-specific helper macros defined locally in
//    scaled_dot_product_attention.metal. Defining them BEFORE the #include
//    of the .metal file means each macro's redefinition inside the .metal
//    body is preceded by ours; xcrun metal's preprocessor will warn but
//    keep the LATER definition. Since the body's helpers ultimately call
//    `instantiate_kernel(...)` (which is no-op'd here), the net AOT-output
//    is empty either way.
#undef  instantiate_kernel
#define instantiate_kernel(...)
#define instantiate_sdpa_vector(...)
#define instantiate_sdpa_vector_aggregation(...)
#define instantiate_sdpa_vector_heads(...)

// 3. Include the actual .metal file body. Its #includes are header-guarded.
#include "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
