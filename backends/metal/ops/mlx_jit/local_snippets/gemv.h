/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Local thin-wrapper snippet for MLX 0.31.2's gemv.metal.
 *
 * MLX's gemv.metal contains both kernel TEMPLATES (GEMVKernel struct +
 * gemv / gemv_t kernel templates we want to JIT-instantiate per-shape)
 * AND ~80 explicit `instantiate_kernel(...)` invocations that compile
 * static AOT instantiations into MLX's own metallib. We only want the
 * templates — the runtime metal compiler will JIT each (BM, BN, SM, SN,
 * TM, TN, dtype, axpby, nc) shape we actually need on demand.
 *
 * Trick: pre-include the headers that gemv.metal would pull in
 * (kernels/utils.h, kernels/steel/utils.h — both #pragma once). This
 * gets defines.h's real `instantiate_kernel` definition into scope.
 * Then #undef + redefine that macro to nothing. When gemv.metal is
 * #included, its own #includes are no-ops (header-guarded), so the
 * macro stays no-op'd through the file body — the AOT instantiations
 * at the bottom expand to nothing while the kernel templates remain.
 *
 * Vendored via the `add_mlx_jit_snippet` machinery in MlxJit.cmake
 * (entry "gemv|...local_snippets/gemv.h"). The resulting
 * Snippets::gemv() string contains GEMVKernel + gemv / gemv_t kernel
 * templates plus the transitive #includes from kernels/utils.h and
 * kernels/steel/utils.h.
 */
#pragma once

// 1. Pull in the shared MLX headers gemv.metal needs (also brings in
//    defines.h's `instantiate_kernel` macro). Both are #pragma once.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

// 2. Now suppress the AOT instantiation macros so the body of
//    gemv.metal doesn't generate static [[host_name(...)]] entries
//    for the runtime metal compiler.
#undef  instantiate_kernel
#define instantiate_kernel(...)
#define instantiate_gemv_helper(...)
#define instantiate_gemv_blocks(...)
#define instantiate_gemv_bs_helper(...)
#define instantiate_gemv_bs_blocks(...)
#define instantiate_gemv_t_bs_helper(...)
#define instantiate_gemv_t_bs_blocks(...)

// 3. Pull in the actual kernel templates. Includes inside this file are
//    header-guarded → won't re-define `instantiate_kernel`.
#include "mlx/backend/metal/kernels/gemv.metal"
