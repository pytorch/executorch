/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmark: linear_coopmat vs linear_vec.
//
// Uses test_etvk.test_mm.default which routes to aten.mm.default.
// When mat2 is constant (set_constant(true)), aten.mm prepacks the weight
// and routes through the linear path:
//   - texture3d output  -> linear_vec (Stephen's tiled shader)
//   - buffer output + coop mat device -> linear_coopmat (KHR cooperative matrix)
//
// For each matrix size, runs two variants:
//   vec_tex: mat1=tex3d, mat2=tex3d(constant), out=tex3d -> linear_vec
//   cm_fp32: mat1=buf, mat2=buf(constant), out=buf -> linear_coopmat

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "cm_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;

std::vector<TestCase> generate_test_cases() {
  std::vector<TestCase> test_cases;

  struct LinearConfig {
    int64_t M, K, N;
    std::string name;
  };

  std::vector<LinearConfig> configs = {
      // BERT-like shapes
      {256, 768, 3072, "BERT_FFN_up"},
      {256, 3072, 768, "BERT_FFN_down"},
      {128, 768, 768, "BERT_QKV"},
      // LLM-like shapes (single token)
      {1, 4096, 4096, "LLM_QKV_1tok"},
      {1, 4096, 11008, "LLM_FFN_up_1tok"},
      {1, 11008, 4096, "LLM_FFN_down_1tok"},
      // LLM-like shapes (batch)
      {32, 4096, 4096, "LLM_QKV_32tok"},
      {32, 4096, 11008, "LLM_FFN_up_32tok"},
      // Square stress
      {256, 1024, 1024, "sq_1024"},
      {256, 4096, 4096, "sq_4096"},
  };

  // Variant 1: linear_vec texture3d (baseline)
  // mat2 is constant -> prepacked -> linear_vec path
  for (const auto& cfg : configs) {
    TestCase tc;
    tc.set_name("vec_tex_" + cfg.name);
    tc.set_operator_name("test_etvk.test_mm.default");

    ValueSpec input_A(
        {cfg.M, cfg.K}, vkapi::kFloat, utils::kTexture3D,
        utils::kWidthPacked, DataGenType::RANDOM);
    ValueSpec input_B(
        {cfg.K, cfg.N}, vkapi::kFloat, utils::kTexture3D,
        utils::kWidthPacked, DataGenType::RANDOM);
    input_B.set_constant(true);
    ValueSpec impl_selector = ValueSpec::make_string("default");
    ValueSpec output(
        {cfg.M, cfg.N}, vkapi::kFloat, utils::kTexture3D,
        utils::kWidthPacked, DataGenType::ZEROS);

    tc.add_input_spec(input_A);
    tc.add_input_spec(input_B);
    tc.add_input_spec(impl_selector);
    tc.add_output_spec(output);
    tc.set_abs_tolerance(1e-2f);
    tc.set_rel_tolerance(1e-1f);
    test_cases.push_back(tc);
  }

  // Variant 2: linear_coopmat buffer fp32
  // mat2 is constant + buffer -> prepack with buffer -> linear_coopmat
  for (const auto& cfg : configs) {
    TestCase tc;
    tc.set_name("cm_fp32_" + cfg.name);
    tc.set_operator_name("test_etvk.test_mm.default");

    ValueSpec input_A(
        {cfg.M, cfg.K}, vkapi::kFloat, utils::kBuffer,
        utils::kWidthPacked, DataGenType::RANDOM);
    ValueSpec input_B(
        {cfg.K, cfg.N}, vkapi::kFloat, utils::kBuffer,
        utils::kWidthPacked, DataGenType::RANDOM);
    input_B.set_constant(true);
    ValueSpec impl_selector = ValueSpec::make_string("default");
    ValueSpec output(
        {cfg.M, cfg.N}, vkapi::kFloat, utils::kBuffer,
        utils::kWidthPacked, DataGenType::ZEROS);

    tc.add_input_spec(input_A);
    tc.add_input_spec(input_B);
    tc.add_input_spec(impl_selector);
    tc.add_output_spec(output);
    tc.set_abs_tolerance(5e-1f);
    tc.set_rel_tolerance(5e-1f);
    test_cases.push_back(tc);
  }

  return test_cases;
}

int64_t linear_flops(const TestCase& test_case) {
  if (test_case.empty() || test_case.num_inputs() < 2) return 0;
  const auto& A = test_case.inputs()[0].get_tensor_sizes();
  const auto& B = test_case.inputs()[1].get_tensor_sizes();
  int64_t M = A.at(A.size() - 2);
  int64_t K = A.at(A.size() - 1);
  int64_t N = B.at(B.size() - 1);
  return 2 * M * N * K;
}

static constexpr int64_t kRefLimit = 2048;

void linear_reference(TestCase& test_case) {
  const ValueSpec& A_spec = test_case.inputs().at(0);
  const ValueSpec& B_spec = test_case.inputs().at(1);
  ValueSpec& out_spec = test_case.outputs().at(0);

  const auto& A_sizes = A_spec.get_tensor_sizes();
  const auto& B_sizes = B_spec.get_tensor_sizes();
  int64_t M = A_sizes.at(A_sizes.size() - 2);
  int64_t K = A_sizes.at(A_sizes.size() - 1);
  int64_t N = B_sizes.at(B_sizes.size() - 1);

  if (M > kRefLimit || K > kRefLimit || N > kRefLimit) {
    std::cerr << "Skipping reference for large matrix ("
              << M << "x" << K << "x" << N << ")" << std::endl;
    return;
  }

  const auto& A_f = A_spec.get_float_data();
  const auto& B_f = B_spec.get_float_data();
  auto& ref = out_spec.get_ref_float_data();
  ref.resize(M * N, 0.0f);

  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        sum += A_f[m * K + k] * B_f[k * N + n];
      ref[m * N + n] = sum;
    }
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  set_print_output(false);
  set_print_latencies(true);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Linear Coopmat vs Vec Microbenchmark" << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan: " << e.what() << std::endl;
    return 1;
  }

  if (api::context()->adapter_ptr()->supports_cooperative_matrix()) {
    std::cout << "Cooperative matrix: SUPPORTED" << std::endl;
    queryCooperativeMatrixProperties();
  } else {
    std::cout << "Cooperative matrix: NOT supported (buffer tests will use linear_vec)" << std::endl;
  }

  auto results = execute_test_cases(
      generate_test_cases,
      linear_flops,
      "LINEAR_COOPMAT_BENCH",
      3,   // warmup
      10,  // benchmark runs
      linear_reference);

  return 0;
}
