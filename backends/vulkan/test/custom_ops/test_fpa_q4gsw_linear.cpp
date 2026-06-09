// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 300;

// Linear configuration struct.
struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size;
  bool has_bias = false;
};

// Convert a ValueSpec's input data (float or half) into a flat
// std::vector<float> for use in the reference implementation.
static std::vector<float> input_to_float_vec(const ValueSpec& spec) {
  if (spec.dtype == vkapi::kFloat) {
    return spec.get_float_data();
  }
  if (spec.dtype == vkapi::kHalf) {
    const auto& half_data = spec.get_half_data();
    std::vector<float> out(half_data.size());
    for (size_t i = 0; i < half_data.size(); ++i) {
      out[i] = half_to_float(half_data[i]);
    }
    return out;
  }
  throw std::invalid_argument(
      "Reference implementation supports only float/half input dtypes.");
}

// Create a single test case for the test_fpa_q4gsw_linear.{gemm,gemv} op.
TestCase create_test_case(
    const LinearConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage,
    int32_t impl_selector,
    bool is_gemv) {
  TestCase test_case;

  const int64_t M = config.M;
  const int64_t K = config.K;
  const int64_t N = config.N;
  const int64_t group_size = config.group_size;

  const bool is_performance =
      (M > kRefDimSizeLimit || K > kRefDimSizeLimit || N > kRefDimSizeLimit);
  const std::string prefix = is_performance ? "PERF" : "ACCU";

  const std::string dtype_str = dtype_short(dtype);
  const std::string shape_str = shape_bracket({M, K}) + "x[" +
      std::to_string(N) + "," + std::to_string(K) + "] g" +
      std::to_string(group_size);
  const std::string storage_str = repr_str(storage, utils::kWidthPacked);
  std::string suffix = std::string("[") + (is_gemv ? "gemv" : "gemm") + " s" +
      std::to_string(impl_selector) + "]";
  suffix += config.has_bias ? " bias" : " no_bias";
  const std::string test_name = make_test_label(
      prefix, dtype_str, dtype_str, shape_str, storage_str, suffix);
  test_case.set_name(test_name);

  const std::string op_name = is_gemv ? "test_etvk.test_fpa_q4gsw_linear.gemv"
                                      : "test_etvk.test_fpa_q4gsw_linear.gemm";
  test_case.set_operator_name(op_name);

  // Input: [M, K]
  ValueSpec input(
      {M, K}, dtype, storage, utils::kWidthPacked, DataGenType::RANDINT);

  // Weight: [N, K/2] uint8 packed 4-bit
  ValueSpec weight(
      {N, K / 2},
      vkapi::kByte,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  weight.set_constant(true);
  weight.set_int4(true);

  // Scales: [K/gs, N] matching input dtype (the custom op prepacks scales
  // using the input tensor's dtype).
  ValueSpec scales(
      {K / group_size, N},
      dtype,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  scales.set_constant(true);

  // Group size
  ValueSpec gs_spec(static_cast<int32_t>(group_size));

  // Bias
  ValueSpec bias(
      {N},
      dtype,
      storage,
      utils::kWidthPacked,
      config.has_bias ? DataGenType::RANDOM : DataGenType::ZEROS);
  bias.set_constant(true);
  if (!config.has_bias) {
    bias.set_none(true);
  }

  // impl_selector as int32
  ValueSpec impl_selector_spec(static_cast<int32_t>(impl_selector));

  // Output: [M, N]
  ValueSpec output(
      {M, N}, dtype, storage, utils::kWidthPacked, DataGenType::ZEROS);

  // Tolerance: fp16 outputs use relaxed tolerance to account for f16
  // accumulation / rounding.
  float base_tol = 0.05f * (static_cast<float>(K) / 64.0f);
  float tol = (dtype == vkapi::kHalf) ? (4.0f * base_tol) : base_tol;
  test_case.set_abs_tolerance(tol);

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);
  test_case.add_input_spec(scales);
  test_case.add_input_spec(gs_spec);
  test_case.add_input_spec(bias);
  test_case.add_input_spec(impl_selector_spec);
  test_case.add_output_spec(output);

  return test_case;
}

// Reference implementation: simple dequant + fp32 GEMM. Only runs for
// small shapes (gate on kRefDimSizeLimit).
void linear_q4gsw_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& scales_spec = test_case.inputs()[idx++];
  const ValueSpec& gs_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];
  // impl_selector is not used in the reference impl
  ++idx;

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();
  auto output_sizes = output_spec.get_tensor_sizes();

  int64_t M = input_sizes[0];
  int64_t K = input_sizes[1];
  int64_t N = output_sizes[1];
  int64_t group_size = gs_spec.get_int_value();

  if (M > kRefDimSizeLimit || K > kRefDimSizeLimit || N > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "Dimensions exceed limit for reference implementation.");
  }

  std::vector<float> input_data = input_to_float_vec(input_spec);
  auto& weight_data = weight_spec.get_uint8_data();
  std::vector<float> scales_data = input_to_float_vec(scales_spec);
  std::vector<float> bias_data;
  if (!bias_spec.is_none()) {
    bias_data = input_to_float_vec(bias_spec);
  }

  int64_t num_output_elements = M * N;
  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float input_val = input_data[m * K + k];

        int64_t weight_idx = n * (K / 2) + (k / 2);
        uint8_t packed = weight_data[weight_idx];
        int8_t nibble = (k % 2 == 0)
            ? static_cast<int8_t>(packed & 0x0F) - 8
            : static_cast<int8_t>((packed >> 4) & 0x0F) - 8;

        int64_t group_idx = k / group_size;
        float scale = scales_data[group_idx * N + n];

        sum += input_val * static_cast<float>(nibble) * scale;
      }
      if (!bias_spec.is_none()) {
        sum += bias_data[n];
      }
      ref_data[m * N + n] = sum;
    }
  }
}

void reference_impl(TestCase& test_case) {
  linear_q4gsw_reference_impl(test_case);
}

// Custom FLOP calculator: 2 * M * K * N for the linear op itself.
int64_t linear_flop_calculator(const TestCase& test_case) {
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t M = input_sizes[0];
  int64_t K = input_sizes[1];
  int64_t N = output_sizes[1];
  return 2 * M * K * N;
}

// Canonical N/K shapes for LLM hidden-size sweeps.
static const std::vector<std::pair<int64_t, int64_t>>& get_nk_shapes() {
  static const std::vector<std::pair<int64_t, int64_t>> kShapes = {
      // (K, N)
      {1024, 2048},
      {4096, 4096},
      // {4096, 14336}, // Large-N case can make the full benchmark binary
      // unstable.
  };
  return kShapes;
}

// GEMM sweep test cases: M in {32, 128, 256} x N/K shapes x dtype x storage
// x impl_selector in kGemmSelectors.
//
// Selector 3 is the legacy in-prod q4gsw linear path
// (et_vk.linear_q4gsw.default) registered in QuantizedLinear.cpp.
std::vector<TestCase> generate_gemm_test_cases() {
  std::vector<TestCase> test_cases;

  const std::vector<int64_t> gemm_Ms = {32, 128, 256};
  const int64_t group_size = 32;

  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
  const std::vector<utils::StorageType> storages = {
      utils::kBuffer, utils::kTexture3D};

  // Selectors exercised in the GEMM PERF/ACCU sweep: PROD (0), forced non-tin
  // / tin GEMM (1, 2), legacy (3).
  const std::vector<int32_t> kGemmSelectors = {0, 1, 2, 3};

  for (int64_t M : gemm_Ms) {
    for (const auto& shape : get_nk_shapes()) {
      const int64_t K = shape.first;
      const int64_t N = shape.second;
      LinearConfig cfg{M, K, N, group_size};
      for (auto dtype : dtypes) {
        for (auto storage : storages) {
          for (int32_t selector : kGemmSelectors) {
            test_cases.push_back(create_test_case(
                cfg, dtype, storage, selector, /*is_gemv=*/false));
          }
        }
      }
    }
  }

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
  for (const auto& shape : kNonAlignedNkShapes) {
    const int64_t K = shape.first;
    const int64_t N = shape.second;
    LinearConfig cfg{32, K, N, group_size};
    for (auto dtype : dtypes) {
      for (auto storage : storages) {
        for (int32_t selector : kNonAlignedSelectors) {
          test_cases.push_back(create_test_case(
              cfg, dtype, storage, selector, /*is_gemv=*/false));
        }
      }
    }
  }

  // Small ACCU shape (M=32, K=128, N=128) under kRefDimSizeLimit so the
  // reference impl runs. Sanity-checks GEMM correctness during iteration.
  {
    LinearConfig cfg{32, 128, 128, group_size};
    for (auto dtype : {vkapi::kFloat, vkapi::kHalf}) {
      test_cases.push_back(create_test_case(
          cfg,
          dtype,
          utils::kTexture3D,
          /*impl_selector=*/0,
          /*is_gemv=*/false));
    }
  }

  // M-tail ACCU shapes. These exercise final partial GEMM tiles for both the
  // fp32 direct-input path (tile height 4) and the fp16 TIN path (tile height
  // 8).
  for (int64_t M : {31, 33}) {
    LinearConfig cfg{M, 128, 128, group_size};
    for (auto dtype : {vkapi::kFloat, vkapi::kHalf}) {
      test_cases.push_back(create_test_case(
          cfg,
          dtype,
          utils::kTexture3D,
          /*impl_selector=*/0,
          /*is_gemv=*/false));
    }
  }

  return test_cases;
}

// GEMV sweep test cases: M = 1 x N/K shapes x dtype x storage x
// impl_selector in {0, 1, 2, 3}.
//
// Selector 3 is the legacy in-prod q4gsw linear path
// (et_vk.linear_q4gsw.default) registered in QuantizedLinear.cpp.
std::vector<TestCase> generate_gemv_test_cases() {
  std::vector<TestCase> test_cases;

  const int64_t group_size = 32;

  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
  const std::vector<utils::StorageType> storages = {
      utils::kBuffer, utils::kTexture3D};

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
  for (const auto& shape : kAccuShapes) {
    const int64_t K = shape.first;
    const int64_t N = shape.second;
    LinearConfig cfg{1, K, N, group_size};
    for (auto dtype : dtypes) {
      for (auto storage : storages) {
        for (int32_t selector : kAccuSelectors) {
          test_cases.push_back(create_test_case(
              cfg, dtype, storage, selector, /*is_gemv=*/true));
        }
      }
    }
  }

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
  for (const auto& shape : kCoopForcedAccuShapes) {
    const int64_t K = shape.first;
    const int64_t N = shape.second;
    LinearConfig cfg{1, K, N, group_size};
    for (auto dtype : dtypes) {
      for (auto storage : storages) {
        for (int32_t selector : kCoopForcedSelectors) {
          test_cases.push_back(create_test_case(
              cfg, dtype, storage, selector, /*is_gemv=*/true));
        }
      }
    }
  }

  // GEMV PERF selectors: PROD (0), forced sg (1), forced nosg (2), LEGACY (3),
  // nc-Buffer coop (13, reuses production prepack — single-format
  // prefill+decode).
  const std::vector<int32_t> kGemvPerfSelectors = {0, 1, 2, 3, 13};
  for (const auto& shape : get_nk_shapes()) {
    const int64_t K = shape.first;
    const int64_t N = shape.second;
    LinearConfig cfg{1, K, N, group_size};
    for (auto dtype : dtypes) {
      for (auto storage : storages) {
        for (int32_t selector : kGemvPerfSelectors) {
          test_cases.push_back(create_test_case(
              cfg, dtype, storage, selector, /*is_gemv=*/true));
        }
      }
    }
  }

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
  for (const auto& shape : kLlmGemvShapes) {
    const int64_t K = shape.first;
    const int64_t N = shape.second;
    LinearConfig cfg{1, K, N, group_size};
    for (int32_t selector : kGemvPerfSelectors) {
      test_cases.push_back(create_test_case(
          cfg, vkapi::kHalf, utils::kTexture3D, selector, /*is_gemv=*/true));
    }
  }
  // Diversity sanity check: K=2048,N=2048 under fp32 + {Tex3D, Buffer} to
  // confirm the regression isn't fp16-Tex3D-specific.
  {
    LinearConfig cfg{1, 2048, 2048, group_size};
    for (auto storage : {utils::kTexture3D, utils::kBuffer}) {
      for (int32_t selector : kGemvPerfSelectors) {
        test_cases.push_back(create_test_case(
            cfg, vkapi::kFloat, storage, selector, /*is_gemv=*/true));
      }
    }
  }

  return test_cases;
}

std::vector<TestCase> generate_all_test_cases() {
  auto gemv = generate_gemv_test_cases();
  auto gemm = generate_gemm_test_cases();
  gemv.insert(gemv.end(), gemm.begin(), gemm.end());
  return gemv;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "FPA Q4GSW Linear A/B Variant Prototyping Framework (gemm + gemv)"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_all_test_cases,
      linear_flop_calculator,
      "FpaQ4gswLinear",
      3,
      10,
      ref_fn);

  return 0;
}
