// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q4gsw_linear/q4gsw_linear.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<LinearConfig>& configs,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& input_dtypes,
    const std::vector<std::string>& ops,
    bool include_narrow_tile_cases = false) {
  std::vector<TestCase> test_cases;

  const bool supports_int8_dot_product =
      vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product();

  for (auto config : configs) {
    std::string prefix =
        (config.M < kRefDimSizeLimit && config.K < kRefDimSizeLimit &&
         config.N < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix + std::to_string(config.M) +
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N) + "_g" +
        std::to_string(config.group_size);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      for (const auto& input_dtype : input_dtypes) {
        for (const auto& op : ops) {
          if (is_dq8ca(op) && !supports_int8_dot_product) {
            continue;
          }
          LinearConfig op_config = config;
          op_config.op_name = op;
          test_cases.push_back(create_test_case_from_config(
              op_config, storage_type, input_dtype));
        }
      }
    }
  }

  if (include_narrow_tile_cases) {
    for (int64_t M : {2, 3, 5}) {
      for (bool has_bias : {true, false}) {
        LinearConfig config;
        config.M = M;
        config.K = 64;
        config.N = 32;
        config.group_size = 32;
        config.has_bias = has_bias;
        config.op_name = "linear_dq8ca_q4gsw";
        config.test_case_name = "correctness_tiledm2_M" + std::to_string(M) +
            (has_bias ? "" : "_no_bias");

        for (const auto& storage_type : storage_types) {
          for (const auto& input_dtype : input_dtypes) {
            TestCase test_case = create_test_case_from_config(
                config, storage_type, input_dtype);
            test_case.set_name(test_case.name() + " [tiledm2]");
            test_case.set_force_narrow_int4_tile(true);
            test_cases.push_back(test_case);
          }
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "correctness") {
  return {
      generate_test_cases(
          {
              // // Gemv test cases
              // {1, 128, 64, 32},
              // {1, 256, 128, 64},
              // Gemm
              {4, 64, 32, 16},
              {4, 128, 64, 32},
              {4, 256, 128, 64},
              {32, 64, 32, 16},
              {32, 128, 64, 32},
              {32, 256, 128, 64},
              {256, 128, 128, 32}, // M=256 > K=128; all dims < kRefDimSizeLimit

              // Coopmat-eligible correctness shapes (M%64==0, N%64==0,
              // K%32==0, group_size%32==0). The Buffer+Half variant fires
              // linear_q4gsw_coopmat / linear_dq8ca_q4gsw_coopmat and is
              // validated against the CPU reference.
              {64, 64, 64, 64},
              {64, 128, 64, 64},
              {64, 256, 128, 128},
              // With bias
              {4, 64, 32, 16, true},
              {4, 128, 64, 32, true},
              {32, 128, 64, 32, true},
              // NOTE: coopmat correctness coverage is NOT in this list. The
              // coopmat dispatch gate requires M%64==0, N%64==0, K%32==0; the
              // smallest qualifying shape (M=64, K=64, N=64) produces enough
              // cancellation outputs that fp16 accumulation drift exceeds any
              // reasonable tolerance against the fp32 reference. Validating
              // the coopmat shader needs a different strategy (e.g.
              // positive-only inputs, or simulating fp16 accumulation in the
              // reference).
          },
          /*storage_types=*/{utils::kTexture3D, utils::kBuffer},
          // Both fp32 and fp16 activations, so the test covers the _float and
          // _half SPIR-V variants of each linear shader. Llama-on-Vulkan
          // exports run with backend.vulkan.force_fp16=True, so the _half
          // variants are the ones we actually hit in production.
          /*input_dtypes=*/{vkapi::kFloat, vkapi::kHalf},
          // Activation+weight quantized and weight-only quantized.
          /*ops=*/{"linear_dq8ca_q4gsw", "linear_q4gsw"},
          /*include_narrow_tile_cases=*/true),
      reference_impl,
      quantized_linear_flop_calculator};
}

REGISTER_TEST_CASE_SET("q4gsw_linear", "performance") {
  return {
      generate_test_cases(
          {
              // A couple of representative performance shapes
              // (coopmat-eligible, M % 64 == 0). The full Llama 3.1 8B prefill
              // sweep lived here during the study; trimmed to keep this a fast
              // unit test.
              {128, 2048, 2048, 128},
              {1024, 4096, 4096, 128},
          },
          /*storage_types=*/{utils::kTexture3D, utils::kBuffer},
          // fp16 is the production path (backend.vulkan.force_fp16=True); fp32
          // covers the _float shader variants.
          /*input_dtypes=*/{vkapi::kFloat, vkapi::kHalf},
          // Activation+weight quantized and weight-only quantized.
          /*ops=*/{"linear_dq8ca_q4gsw", "linear_q4gsw"}),
      reference_impl,
      quantized_linear_flop_calculator};
}

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
