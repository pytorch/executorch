// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

std::vector<TestCase> generate_gated_delta_rule_test_cases() {
  std::vector<TestCase> test_cases;

  // Set the data generation type
  DataGenType data_gen_type = DataGenType::ONES;

  std::vector<std::vector<int64_t>> size_configs = {
      {1, 2, 4, 32}, // Batch=1, Heads=2, Seq=4, Dim=32 to prevent inf overflow
  };

  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};
  std::vector<vkapi::ScalarType> data_types = {vkapi::kFloat};

  for (const auto& sizes : size_configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        TestCase test_case;

        std::string test_name = "GATED_DELTA_RULE_TEST";
        test_case.set_name(test_name);
        test_case.set_operator_name("llama.gated_delta_rule.default");

        int64_t B = sizes[0];
        int64_t H = sizes[1];
        int64_t Seq = sizes[2];
        int64_t Dim = sizes[3];
        
        // q, k are [B, Seq, H, Dim] (Wait, MLX says [B, T, Hk, Dk]. We'll use [B, Seq, H, Dim])
        std::vector<int64_t> qkv_sizes = {B, Seq, H, Dim};

        // q, k, v, decay
        ValueSpec q(qkv_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        ValueSpec k(qkv_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        ValueSpec v(qkv_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        
        // decay and beta are [B, Seq, H]
        std::vector<int64_t> gate_sizes = {B, Seq, H};
        ValueSpec decay(gate_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        ValueSpec beta(gate_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        
        // initial_state is [B, H, Dv, Dk] = [B, H, Dim, Dim]
        std::vector<int64_t> state_sizes = {B, H, Dim, Dim};
        ValueSpec initial_state(state_sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);

        // Outputs
        ValueSpec out(qkv_sizes, data_type, storage_type, utils::kWidthPacked, DataGenType::ZEROS);
        ValueSpec final_state(state_sizes, data_type, storage_type, utils::kWidthPacked, DataGenType::ZEROS);

        test_case.add_input_spec(q);
        test_case.add_input_spec(k);
        test_case.add_input_spec(v);
        test_case.add_input_spec(decay);
        test_case.add_input_spec(beta);
        test_case.add_input_spec(initial_state);

        test_case.add_output_spec(out);
        test_case.add_output_spec(final_state);

        test_cases.push_back(test_case);
      }
    }
  }

  return test_cases;
}

int64_t gated_delta_rule_flop_calculator(const TestCase& test_case) {
  return 100; // Dummy FLOPs
}

void dummy_reference_compute(TestCase& test_case) {
  const ValueSpec& q_spec = test_case.inputs().at(0);
  const ValueSpec& k_spec = test_case.inputs().at(1);
  const ValueSpec& v_spec = test_case.inputs().at(2);
  const ValueSpec& decay_spec = test_case.inputs().at(3);
  const ValueSpec& beta_spec = test_case.inputs().at(4);
  const ValueSpec& state_spec = test_case.inputs().at(5);
  
  ValueSpec& out_spec = test_case.outputs().at(0);
  ValueSpec& final_state_spec = test_case.outputs().at(1);
  
  auto& q = q_spec.get_float_data();
  auto& k = k_spec.get_float_data();
  auto& v = v_spec.get_float_data();
  auto& decay = decay_spec.get_float_data();
  auto& beta = beta_spec.get_float_data();
  auto& initial_state = state_spec.get_float_data();
  
  auto& out = out_spec.get_ref_float_data();
  out.resize(out_spec.numel());
  
  auto& final_state = final_state_spec.get_ref_float_data();
  final_state.resize(final_state_spec.numel());
  
  for (size_t i = 0; i < state_spec.numel(); ++i) {
      final_state[i] = initial_state[i];
  }
  
  int B = q_spec.sizes[0];
  int Seq = q_spec.sizes[1];
  int H = q_spec.sizes[2];
  int Dim = q_spec.sizes[3];
  
  for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
          for (int t = 0; t < Seq; ++t) {
              float g_t = decay[b * Seq * H + t * H + h];
              float b_t = beta[b * Seq * H + t * H + h];
              
              std::vector<float> kv_mem(Dim, 0.0f);
              for (int d_v = 0; d_v < Dim; ++d_v) {
                  for (int d_k = 0; d_k < Dim; ++d_k) {
                      int state_idx = b * (H * Dim * Dim) + h * (Dim * Dim) + d_v * Dim + d_k;
                      final_state[state_idx] *= g_t;
                      
                      int k_idx = b * (Seq * H * Dim) + t * (H * Dim) + h * Dim + d_k;
                      kv_mem[d_v] += final_state[state_idx] * k[k_idx];
                  }
              }
              
              std::vector<float> delta(Dim, 0.0f);
              for (int d_v = 0; d_v < Dim; ++d_v) {
                  int v_idx = b * (Seq * H * Dim) + t * (H * Dim) + h * Dim + d_v;
                  delta[d_v] = (v[v_idx] - kv_mem[d_v]) * b_t;
              }
              
              std::vector<float> y_t(Dim, 0.0f);
              for (int d_v = 0; d_v < Dim; ++d_v) {
                  for (int d_k = 0; d_k < Dim; ++d_k) {
                      int state_idx = b * (H * Dim * Dim) + h * (Dim * Dim) + d_v * Dim + d_k;
                      int k_idx = b * (Seq * H * Dim) + t * (H * Dim) + h * Dim + d_k;
                      int q_idx = b * (Seq * H * Dim) + t * (H * Dim) + h * Dim + d_k;
                      
                      final_state[state_idx] += k[k_idx] * delta[d_v];
                      y_t[d_v] += final_state[state_idx] * q[q_idx];
                  }
                  int out_idx = b * (Seq * H * Dim) + t * (H * Dim) + h * Dim + d_v;
                  out[out_idx] = y_t[d_v];
              }
          }
      }
  }
}

int main(int argc, char* argv[]) {
  set_print_output(false); 
  set_print_latencies(true);
  set_use_gpu_timestamps(true); 

  std::cout << "Gated Delta Rule Prototyping Test" << std::endl;

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what() << std::endl;
    return 1;
  }

  auto results = execute_test_cases(
      generate_gated_delta_rule_test_cases,
      gated_delta_rule_flop_calculator,
      "GatedDeltaRule",
      1,
      1,
      dummy_reference_compute);

  return 0;
}
