/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MLX JIT Snippets — declarations of per-snippet MSL source getters.
// Each function returns a pointer to a C string containing the JIT preamble
// for a single MLX `make_jit_source` boundary (mirrors mlx 0.31.2's
// `mlx::core::metal::*` overloads in jit_kernels.cpp).
// IMPLEMENTATIONS are auto-generated at CMake configure time by
// make_mlx_jit_snippet.sh from the MLX 0.31.2 submodule headers under
//   third-party/executorch/backends/mlx/third-party/mlx/
//     mlx/backend/metal/kernels/...
// The generated .cpp files live under
//   ${CMAKE_BINARY_DIR}/mlx_jit_snippets/
// and are compiled into metal_backend like any other source.
// To add a new snippet:
//   1. Add a declaration here.
//   2. Add a corresponding `add_mlx_jit_snippet(<subpath>)` call in
//      backends/apple/metal/CMakeLists.txt (passing the path under
//      kernels/, without `.h`). The CMake macro derives the function name
//      via basename, so steel/gemm/gemm → gemm().
//===----------------------------------------------------------------------===//

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit {
namespace Snippets {

// kernels/utils.h — top-level shared header (bf16, complex, defines,
// logging, type-limits, WorkPerThread, etc.). Maps to MLX's
// `mlx::core::metal::utils()`. NOT to be confused with steel/utils.h
// which is included transitively via gemm() / gemm_nax().
const char* utils();

// kernels/steel/gemm/gemm.h — pulls in steel utils, loader, mma, params,
// transforms (the SIMD-MMA gemm helpers, including the `gemm_loop` template).
const char* gemm();

// kernels/steel/gemm/kernels/steel_gemm_fused.h — defines the fused
// dense-GEMM kernel template (SIMD-MMA path, used by Apple7-Apple8 +
// Apple9 fallback). Pairs with gemm() + utils().
const char* steel_gemm_fused();

// kernels/steel/gemm/kernels/steel_gemm_splitk.h — defines split-K
// SIMD-MMA partial + accum kernel templates. Pairs with gemm() + utils().
const char* steel_gemm_splitk();

// kernels/steel/gemm/gemm_nax.h — pulls in steel utils, nax (NAX
// MMA helpers), params, transforms. Used by all NAX kernels.
const char* gemm_nax();

// kernels/steel/gemm/kernels/steel_gemm_fused_nax.h — fused dense-GEMM
// using NAX cooperative-tensor matmul (Apple9+ only). Pairs with
// gemm_nax() + utils().
const char* steel_gemm_fused_nax();

// kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h — split-K NAX
// partial kernel. The accum kernel is the same as the SIMD path
// (steel_gemm_splitk's accum). Pairs with gemm_nax() + utils().
const char* steel_gemm_splitk_nax();

// gemv kernels (vendored from MLX 0.31.2's gemv.metal via a thin local
// wrapper that suppresses the AOT instantiation macros so we can JIT
// each (BM, BN, SM, SN, TM, TN, dtype, axpby, nc) shape on demand).
// Includes both gemv and gemv_t kernel templates plus the GEMVKernel
// struct that backs them.
const char* gemv();

// kernels/scaled_dot_product_attention.metal — vendored via a thin
// local_snippets/sdpa_vector.h wrapper that suppresses the AOT
// `instantiate_sdpa_vector_heads(...)` calls. Provides the templates:
//   sdpa_vector<T, D, V>           — single-pass vector decode
//   sdpa_vector_2pass_1<T, D, V>   — 2-pass partial (long-kL decode)
//   sdpa_vector_2pass_2<T, D>      — 2-pass aggregation
// Plus the function-constant slots 20-26 (has_mask, query_transposed,
// do_causal, bool_mask, float_mask, has_sinks, blocks).
const char* sdpa_vector();

// kernels/steel/attn/kernels/steel_attention.metal — vendored via a thin
// local_snippets/steel_attention.h wrapper. Provides the SIMD-MMA
// "steel" attention template (Apple7-Apple8; Apple9 fallback for
// D==80 / fp32-no-tf32):
//   attention<T, BQ, BK, BD, WM, WN, MaskType, AccumType=float>
// Function constants 200/201/300/301/302 (align_Q, align_K, has_mask,
// do_causal, has_sinks). Brings in the AttnParams / AttnMaskParams
// structs from steel/attn/params.h transitively.
const char* steel_attention();

// kernels/steel/attn/kernels/steel_attention_nax.metal — NAX
// cooperative-tensor variant of steel_attention (Apple9+). Same
// template signature (`attention_nax<...>`) and FCs as steel_attention
// but uses the NAX MMA helpers from steel/attn/nax.h.
const char* steel_attention_nax();

// kernels/quantized.metal — vendored via local_snippets/quantized.h
// wrapper. Provides MLX's affine-quantized linear kernel templates:
//   affine_qmv_fast<T, group_size, bits, batched>      — decode fast path
//   affine_qmv<T, group_size, bits, batched>           — decode generic
//   affine_qmv_quad<T, group_size, bits, D, batched>   — decode quad path (D∈{64,128})
//   affine_qmm_t<T, group_size, bits, aligned_N, BM=32, BK=32, BN=32>
//                                                      — prefill, weight-transposed
//   affine_qvm<...>, affine_qmm_n<...>, affine_qmm_t_splitk<...>, ...
// Paired with utils() (transitively included). NO function constants —
// alignment + symmetric/asym are baked into the kernel name & template.
const char* quantized();

// kernels/quantized_nax.metal — NAX cooperative-tensor variants of the
// affine-quantized kernels (Apple9+). Same template families but using
// NAX MMA helpers from steel/gemm/nax.h.
const char* quantized_nax();

} // namespace Snippets
} // namespace mlx_jit
} // namespace metal_v2
} // namespace backends
} // namespace executorch
