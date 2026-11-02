// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q4gsw_linear/q4gsw_linear.h>

#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

namespace {

// One case per M x (K, N) x dtype x storage x impl_selector, in that nesting
// order.
struct FpaSweep {
  std::vector<int64_t> Ms;
  std::vector<std::pair<int64_t, int64_t>> nk_shapes; // (K, N)
  std::vector<vkapi::ScalarType> dtypes;
  std::vector<utils::StorageType> storages;
  std::vector<int32_t> impl_selectors;
  bool is_gemv;
};

std::vector<TestCase> generate_fpa_test_cases(
    int64_t group_size,
    const std::vector<FpaSweep>& sweeps) {
  std::vector<TestCase> test_cases;
  for (const auto& sweep : sweeps) {
    for (int64_t M : sweep.Ms) {
      for (const auto& shape : sweep.nk_shapes) {
        const int64_t K = shape.first;
        const int64_t N = shape.second;
        FpaLinearConfig cfg{M, K, N, group_size};
        for (auto dtype : sweep.dtypes) {
          for (auto storage : sweep.storages) {
            for (int32_t selector : sweep.impl_selectors) {
              test_cases.push_back(create_test_case(
                  cfg, dtype, storage, selector, sweep.is_gemv));
            }
          }
        }
      }
    }
  }
  return test_cases;
}

const std::vector<vkapi::ScalarType> kDtypes = {vkapi::kFloat, vkapi::kHalf};
const std::vector<utils::StorageType> kStorages = {
    utils::kBuffer,
    utils::kTexture3D};

// ACCU correctness shapes (under kRefDimSizeLimit=300). Exercise selectors
// PROD (0) and forced nosg (2). Forced sg (1) is intentionally skipped:
// sg requires subgroupSize==64 and produces incorrect results on Mali
// (subgroupSize==16); on those devices the PROD picker correctly routes
// to the nosg variant. N must be a multiple of 128 (= 2 * LWG.x) so the
// GEMV shader has no early-exit threads in any workgroup.
const std::vector<std::pair<int64_t, int64_t>> kAccuShapes = {
    // (K, N)
    {128, 128},
    {256, 256},
};
// Selector 13 (nc Buffer, reuses production prepack) included for ACCU
// coverage of the coop nc weight-binding variant.
const std::vector<int32_t> kAccuSelectors = {0, 2, 13};

// Forced coop-reduction-decomposition ACCU coverage (selectors 14/15/16 =
// g1w64 / g4w16 / g8w8). The production picker (pick_coop_variant_for_N)
// only selects g4w16 (1024<N<=4096) and g8w8 (N>4096) at PERF-sized N, where
// every dim exceeds kRefDimSizeLimit=300 so the reference impl is skipped and
// those reduction decompositions get zero numeric validation. These cases
// pin each decomposition regardless of N at small shapes (all dims <= 300)
// so the reference runs and proves g4w16 / g8w8 compute the SAME result as
// the reference — the M=1 decode path actually shipped for Qwen3 / Llama.
//
// Each WG of variant gN produces N*8 outputs (g1w64 -> 8, g4w16 -> 32,
// g8w8 -> 64), so the N values below tile cleanly into all three: N=128
// (16 / 4 / 2 WGs) and N=256 (32 / 8 / 4 WGs). The shader also handles a
// ragged final WG, but clean tiles keep the test intent unambiguous. K is
// swept over {64, 128, 256} (all multiples of group_size=32 and <= 300) so
// the K-loop reduction is exercised across short and longer accumulations.
const std::vector<std::pair<int64_t, int64_t>> kCoopForcedAccuShapes = {
    // (K, N) — all dims <= 300 so linear_q4gsw_reference_impl runs.
    {64, 128},
    {128, 256},
    {256, 128},
};
// 14 -> g1w64, 15 -> g4w16, 16 -> g8w8. g1w64 included for symmetry.
const std::vector<int32_t> kCoopForcedSelectors = {14, 15, 16};

// Non-aligned-N coverage for the W_4X8 GEMM path. The fp32 GEMM issues a
// 16B ivec4 weight load that spans two consecutive (k4, n4) ivec2 tiles
// along N, so N4 must be even (== N a multiple of 8) at the buffer-stride
// level. The prepack pads the weight buffer's row stride to next-even N4
// and fills the OOB tiles with bias-zero nibbles; these accuracy cases
// exercise that padding path on shapes with N % 8 != 0. Only the new W_4X8
// family is tested (selectors 0, 1, 2, 5, 6) — selector 3 (LEGACY) uses a
// different prepack and supports arbitrary N.
const std::vector<std::pair<int64_t, int64_t>> kNonAlignedNkShapes = {
    // (K, N) — K kept under kRefDimSizeLimit so reference impl runs.
    {128, 12},
    {128, 20},
};
const std::vector<int32_t> kNonAlignedSelectors = {0, 1, 2};

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "fpa_correctness") {
  return {
      generate_fpa_test_cases(
          /*group_size=*/32,
          {
              // GEMV
              {{1},
               kAccuShapes,
               kDtypes,
               kStorages,
               kAccuSelectors,
               /*is_gemv=*/true},
              {{1},
               kCoopForcedAccuShapes,
               kDtypes,
               kStorages,
               kCoopForcedSelectors,
               /*is_gemv=*/true},
              // GEMM
              {{32},
               kNonAlignedNkShapes,
               kDtypes,
               kStorages,
               kNonAlignedSelectors,
               /*is_gemv=*/false},
              // Small ACCU shape (M=32, K=128, N=128) under kRefDimSizeLimit so
              // the reference impl runs. Sanity-checks GEMM correctness during
              // iteration.
              {{32},
               {{128, 128}},
               kDtypes,
               {utils::kTexture3D},
               /*impl_selectors=*/{0},
               /*is_gemv=*/false},
              // M-tail ACCU shapes. These exercise final partial GEMM tiles for
              // both the fp32 direct-input path (tile height 4) and the fp16
              // TIN path (tile height 8).
              {{31, 33},
               {{128, 128}},
               kDtypes,
               {utils::kTexture3D},
               /*impl_selectors=*/{0},
               /*is_gemv=*/false},
          }),
      fpa_reference_impl,
      linear_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

namespace {

// Canonical N/K shapes for LLM hidden-size sweeps.
const std::vector<std::pair<int64_t, int64_t>> kNkShapes = {
    // (K, N)
    {1024, 2048},
    {4096, 4096},
    // {4096, 14336}, // Large-N case can make the full benchmark binary
    // unstable.
};

// GEMV PERF selectors: PROD (0), forced sg (1), forced nosg (2), LEGACY (3),
// nc-Buffer coop (13, reuses production prepack — single-format
// prefill+decode).
const std::vector<int32_t> kGemvPerfSelectors = {0, 1, 2, 3, 13};

// LLM-decode-shape PERF cells (M=1 GEMV, group_size=32). Mirrors the actual
// per-layer linear shapes seen during decode profiling on Llama 3.2 1B and
// Qwen3 0.6B; the original Phase 2 corpus (1024/2048/4096 x 2048/4096/11008)
// under-samples these and missed the regression where sg-GEMV (selector 1)
// is 15-22% slower per dispatch than LEGACY coop (selector 3) on Adreno 750.
//
// All N values here are multiples of 128 (= 2 * LWG.x for the GEMV shader),
// so the GEMV shader has no early-exit threads. N=512 is the Llama 3.2 1B
// k_proj/v_proj projection (GQA) and is a multiple of 4 (the prepack
// requirement: prepack_q4_w_4x8_nc_buffer enforces N % 4 == 0).
//
// Default storage is fp16 + Tex3D - that's the actual decode config and the
// shape combo where the regression was observed. We additionally exercise
// K=2048,N=2048 under fp32 + Tex3D and fp32 + Buffer to confirm the
// regression isn't fp16-Tex3D-specific. All four selectors (PROD, sg, nosg,
// LEGACY) are exercised.
const std::vector<std::pair<int64_t, int64_t>> kLlmGemvShapes = {
    // (K, N) - Llama 3.2 1B
    {2048, 512}, // k_proj / v_proj (GQA)
    {2048, 2048}, // q_proj
    {2048, 8192}, // gate_proj / up_proj
    {8192, 2048}, // down_proj
    // (K, N) - Qwen3 0.6B
    {1024, 1024}, // k_proj / v_proj
    {1024, 2048}, // q_proj (also overlaps with original corpus)
    {1024, 3072}, // gate_proj / up_proj
    {3072, 1024}, // down_proj
};

// Selectors exercised in the GEMM PERF/ACCU sweep: PROD (0), forced non-tin
// / tin GEMM (1, 2), legacy (3).
//
// Selector 3 is the legacy in-prod q4gsw linear path
// (et_vk.linear_q4gsw.default) registered in QuantizedLinear.cpp.
const std::vector<int32_t> kGemmSelectors = {0, 1, 2, 3};

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "fpa_performance") {
  return {
      generate_fpa_test_cases(
          /*group_size=*/32,
          {
              // GEMV sweep: M = 1 x N/K shapes x dtype x storage x
              // impl_selector.
              {{1},
               kNkShapes,
               kDtypes,
               kStorages,
               kGemvPerfSelectors,
               /*is_gemv=*/true},
              {{1},
               kLlmGemvShapes,
               {vkapi::kHalf},
               {utils::kTexture3D},
               kGemvPerfSelectors,
               /*is_gemv=*/true},
              // Diversity sanity check: K=2048,N=2048 under fp32 + {Tex3D,
              // Buffer} to confirm the regression isn't fp16-Tex3D-specific.
              {{1},
               {{2048, 2048}},
               {vkapi::kFloat},
               {utils::kTexture3D, utils::kBuffer},
               kGemvPerfSelectors,
               /*is_gemv=*/true},
              // GEMM sweep: M in {32, 128, 256} x N/K shapes x dtype x storage
              // x impl_selector.
              {{32, 128, 256},
               kNkShapes,
               kDtypes,
               kStorages,
               kGemmSelectors,
               /*is_gemv=*/false},
          }),
      fpa_reference_impl,
      linear_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
