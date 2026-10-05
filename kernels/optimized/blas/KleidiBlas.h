/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/optimized/blas/CPUBlas.h>

#include <cstddef>
#include <cstdint>

#ifdef ET_BUILD_WITH_KLEIDIAI
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cpuinfo.h>
#include <executorch/runtime/platform/assert.h>

#ifndef ET_KLEIDIAI_DISABLE_NEON_BF16
#if __has_include(<kai/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h>)
#define ET_KLEIDIAI_HAS_NEON_BF16
#include <kai/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h>
#elif __has_include(<kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h>)
#define ET_KLEIDIAI_HAS_NEON_BF16
#include <kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h>
#endif
#endif

#if __has_include( \
    <kai/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h>)
#define ET_KLEIDIAI_HAS_SME2_BF16
#include <kai/kai_lhs_pack_x16p2vlx2_x16_sme.h>
#include <kai/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h>
#include <kai/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h>
#include <kai/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.h>
#elif __has_include(<kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h>)
#define ET_KLEIDIAI_HAS_SME2_BF16
#include <kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h>
#include <kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.h>
#endif
#endif // ET_BUILD_WITH_KLEIDIAI

namespace executorch {
namespace cpublas {

#if defined(ET_KLEIDIAI_HAS_NEON_BF16) || defined(ET_KLEIDIAI_HAS_SME2_BF16)
// Column-major c = beta * c + alpha * (a @ b) via KleidiAI, where a and b are
// BFloat16 and c is float. Returns false when no KleidiAI backend applies, in
// which case the caller falls back to gemm_impl. Defined in KleidiBlas.cpp.
// (gemm_uses_kleidiai_bfloat16 stays declared in CPUBlas.h so existing
// callers keep including only that header; it is also defined here.)
// clang-format off
bool kleidiai_bfloat16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const executorch::aten::BFloat16* a, int64_t lda,
    const executorch::aten::BFloat16* b, int64_t ldb,
    float beta,
    float* c, int64_t ldc);
// clang-format on
#endif

} // namespace cpublas
} // namespace executorch
