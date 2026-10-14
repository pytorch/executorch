// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

constexpr int64_t kRefDimSizeLimit = 300;

// Linear configuration struct
struct LinearConfig {
  int64_t M; // Batch size / number of rows in input
  int64_t K; // Input features / columns in input, rows in weight
  int64_t N; // Output features / columns in weight
  int64_t group_size; // Number of input channels per quantization group
  bool has_bias = false;
  std::string test_case_name = "placeholder";
  std::string op_name = "linear_dq8ca_q4gsw";
};

TestCase create_test_case_from_config(
    const LinearConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype);

void reference_impl(TestCase& test_case);

int64_t quantized_linear_flop_calculator(const TestCase& test_case);

// FPA Q4GSW Linear A/B benchmark binary.
//
// Each generated test case has an `impl_selector` arg routed to the test
// op `test_etvk.test_fpa_q4gsw_linear.{gemm,gemv}` in TestFpaQ4gswLinear.cpp:
//
//   GEMM (is_gemv=false):
//     0  -> PROD                       (et_vk.q4gsw_linear.default; dtype-based
//     picker) 1  -> GEMM_W_4X8                 (forced non-tin GEMM, nc buffer
//     weight) 2  -> GEMM_TIN_W_4X8             (forced tin GEMM, nc buffer
//     weight) 3  -> LEGACY                     (et_vk.linear_q4gsw.default
//     legacy shaders)
//
//   GEMV (is_gemv=true):
//     0  -> PROD                        (et_vk.q4gsw_linear.default;
//     dtype-based picker) 1  -> GEMV_W_4X8                  (forced gemv with
//     subgroup broadcast) 2  -> GEMV_W_4X8_NOSG             (forced gemv
//     without subgroup broadcast) 3  -> LEGACY (et_vk.linear_q4gsw.default
//     legacy shaders) 13 -> GEMV_COOP_W_4X8_NC_BUFFER   (coop GEMV reusing the
//     production
//                                        nc-buffer prepack — same payload as
//                                        W_4X8 GEMM/TIN GEMM/sg-GEMV;
//                                        == g1w64 decomposition)
//     14 -> GEMV_COOP_..._G1W64        (force NUM_GROUPS=1,
//     WORKERS_PER_GROUP=64) 15 -> GEMV_COOP_..._G4W16        (force
//     NUM_GROUPS=4, WORKERS_PER_GROUP=16) 16 -> GEMV_COOP_..._G8W8 (force
//     NUM_GROUPS=8, WORKERS_PER_GROUP=8)
//
// Selectors 14-16 pin the coop nc-buffer GEMV to an explicit reduction
// decomposition regardless of N. The production picker (pick_coop_variant_for_N
// in Q4gswLinear.cpp) only chooses g4w16 / g8w8 at PERF-sized N where the
// reference impl is skipped; these forced selectors give g4w16 / g8w8 numeric
// (ACCU) coverage at small N. Production picker behavior is unchanged.
//
// Selector 3 (LEGACY) is the in-prod q4gsw linear path. It uses a different
// prepack (pack_q4_linear_weight) and shader family
// (linear_q4gsw_tiled_* / linear_q4gsw_coop_*); the framework's per-shader
// timing breakdown will pick those up automatically.

// Linear configuration struct.
struct FpaLinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size;
  bool has_bias = false;
};

TestCase create_test_case(
    const FpaLinearConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage,
    int32_t impl_selector,
    bool is_gemv);

void fpa_reference_impl(TestCase& test_case);

int64_t linear_flop_calculator(const TestCase& test_case);

// Consolidated coopmat-vs-tiled microbenchmark for the two int4-quantized
// linear types at Llama 3.1 8B prefill shapes:
//   4w    = linear_q4gsw          (weight-only int4)
//   8da4w = linear_dq8ca_q4gsw    (dyn-act int8 x int4 weight)
//
// Baseline (tiled) is selected by Texture3D+Half output storage; coopmat is
// selected by Buffer+Half (the runtime gate in QuantizedLinear.cpp picks the
// _coopmat shader when M%64==0, N%64==0, K%32==0, subgroup==64). The CPU
// reference only runs for the small shapes in coopmat_correctness.

struct CoopmatLinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size; // only meaningful for 4-bit
  std::string op_name;
};

bool is_dq8ca(const std::string& op);

bool is_4bit(const std::string& op);

TestCase make_case(const CoopmatLinearConfig& cfg, utils::StorageType storage);

void bench_reference(TestCase& tc);

int64_t flop_calc(const TestCase& tc);

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
