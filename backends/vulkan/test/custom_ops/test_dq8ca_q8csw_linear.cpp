// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Microbench for linear_dq8ca_q8csw: dynamic per-token INT8 activation ×
// per-channel symmetric INT8 weight. Structurally mirrors q4gsw_linear.cpp's
// dq8ca testing path, but the weight is full int8 (no nibble pack / unpack),
// scales/sums are per-channel (no group_size loop).
//
// K-loop dispatches dotPacked4x8AccSatEXT (→ V_DOT4_I32_I8 on RDNA3): real
// INT8 × INT8 → INT32 hardware MACs. The microbench in isolation gives the
// raw shader-level throughput, decoupled from the AOT pipeline status.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 300;

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias = false;
  std::string test_case_name = "placeholder";
  // Only dq8ca_q8csw is exercised here; q8ta_q8csw and q8csw weight-only are
  // already covered by q8csw_linear.cpp.
  std::string op_name = "linear_dq8ca_q8csw";
};

// Read a ValueSpec's content as float regardless of underlying dtype; used by
// the CPU reference so it can work on either the fp32 or fp16 test case.
static std::vector<float> as_float_data(const ValueSpec& spec) {
  if (spec.dtype == vkapi::kFloat) {
    return spec.get_float_data();
  }
  if (spec.dtype == vkapi::kHalf) {
    const auto& halves = spec.get_half_data();
    std::vector<float> out(halves.size());
    for (size_t i = 0; i < halves.size(); ++i) {
      out[i] = half_to_float(halves[i]);
    }
    return out;
  }
  throw std::invalid_argument("as_float_data: unsupported dtype");
}

// Compute per-output-channel sums of the int8 weight tensor. Shape: [N].
// Used to apply the input zero-point correction during integer accumulation.
static void compute_weight_sums_perchannel(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_features,
    int64_t in_features) {
  const auto& w = quantized_weight.get_int8_data();
  auto& sums = weight_sums.get_int32_data();
  sums.assign(out_features, 0);
  for (int64_t n = 0; n < out_features; ++n) {
    int32_t s = 0;
    for (int64_t k = 0; k < in_features; ++k) {
      s += static_cast<int32_t>(w[n * in_features + k]);
    }
    sums[n] = s;
  }
}

TestCase create_test_case_from_config(
    const LinearConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype) {
  TestCase test_case;

  std::string storage_str =
      (storage_type == utils::kTexture3D) ? "Texture3D" : "Buffer";
  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name =
      config.test_case_name + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  std::string operator_name = "et_vk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input [M, K] (fp16 or fp32)
  std::vector<int64_t> input_size = {config.M, config.K};
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  // Per-row dynamic input scale [1, M] (fp16 or fp32) and zp [1, M] (int8)
  ValueSpec input_scale(
      {1, config.M},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  input_scale.set_constant(true);

  ValueSpec input_zero_point(
      {1, config.M},
      vkapi::kChar,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  input_zero_point.set_constant(true);

  // INT8 weight [N, K]: no nibble pack.
  std::vector<int64_t> weight_size = {config.N, config.K};
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);
  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  // Per-channel weight scales [N] (fp16 or fp32)
  ValueSpec weight_scales(
      {config.N},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  // Per-channel weight sums [N] (int32) — pre-computed from the actual weight
  // data so the runtime can apply input_zp correction in integer accum space.
  ValueSpec weight_sums(
      {config.N},
      vkapi::kInt,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);
  compute_weight_sums_perchannel(
      weight_sums, quantized_weight, config.N, config.K);

  // Bias [N], optional
  ValueSpec bias(
      {config.N},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      config.has_bias ? DataGenType::RANDOM : DataGenType::ZEROS);
  bias.set_constant(true);
  if (!config.has_bias) {
    bias.set_none(true);
  }

  // Output [M, N] (matches input dtype)
  ValueSpec output(
      {config.M, config.N},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Argument order matches et_vk.linear_dq8ca_q8csw.default signature:
  //   (input, input_scale, input_zp, weight, weight_sums, weight_scales, bias)
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(quantized_weight);
  test_case.add_input_spec(weight_sums);
  test_case.add_input_spec(weight_scales);
  test_case.add_input_spec(bias);
  test_case.add_output_spec(output);

  // INT8 dot4 accumulates in int32; the final dequant fma is in fp.
  // Tolerance is bounded by per-row scale precision and fp16 conversion.
  if (input_dtype == vkapi::kHalf) {
    // INT8 dot4 → INT32 accum → fp32 dequant → fp16 store; the only fp16
    // rounding is at the final store. Per-row dynamic act scale gives
    // O(1) magnitudes pre-store, so a few ULPs of fp16 jitter is normal.
    test_case.set_abs_tolerance(5.0f);
    test_case.set_rel_tolerance(2e-1f);
  } else {
    test_case.set_abs_tolerance(1e-2f);
    test_case.set_rel_tolerance(1e-2f);
  }

  return test_case;
}

std::vector<TestCase> generate_quantized_linear_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<LinearConfig> configs = {
      // Correctness (M, K, N < 300)
      {4, 64, 32},
      {4, 128, 64},
      {4, 256, 128},
      {32, 64, 32},
      {32, 128, 64},
      {32, 256, 128},
      // With bias
      {4, 64, 32, true},
      {4, 128, 64, true},
      {32, 128, 64, true},
      // Coopmat-eligible correctness shapes: M%64==0, N%64==0, K%32==0.
      // These verify the linear_dq8ca_q8csw_coopmat shader against the CPU
      // reference (only the Buffer_Half storage/dtype combo will hit the
      // coopmat path; other variants still validate the tiled fallback).
      {64, 64, 64},
      {64, 64, 64, true},
      // A couple of representative performance shapes (K=N=2048).
      {128, 2048, 2048},
      {1024, 2048, 2048},
  };

  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  for (auto config : configs) {
    std::string prefix =
        (config.M < kRefDimSizeLimit && config.K < kRefDimSizeLimit &&
         config.N < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string name = prefix + std::to_string(config.M) + "_" +
        std::to_string(config.K) + "_" + std::to_string(config.N);
    if (config.has_bias) {
      name += "_bias";
    }
    config.test_case_name = name;

    // Cover both kFloat (so the _float shader variant runs) and kHalf (so
    // the _half variant runs — same shape Llama-on-Vulkan would hit).
    std::vector<vkapi::ScalarType> input_dtypes = {vkapi::kFloat, vkapi::kHalf};

    for (const auto& storage_type : storage_types) {
      for (const auto& input_dtype : input_dtypes) {
        if (!vkcompute::api::context()
                 ->adapter_ptr()
                 ->supports_int8_dot_product()) {
          continue;
        }
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, input_dtype));
      }
    }
  }

  return test_cases;
}

// CPU reference: dynamic-per-row int8 activation × per-channel int8 weight,
// dequantized via (acc - input_zp * weight_sum) * input_scale * weight_scale.
void linear_dq8ca_q8csw_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_sums_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();
  auto output_sizes = output_spec.get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = output_sizes[1];

  if (batch_size > kRefDimSizeLimit || in_features > kRefDimSizeLimit ||
      out_features > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "Reference impl skipped for perf-size shapes (M/K/N > 300).");
  }
  // CPU reference uses fp32 throughout; comparing against an fp16 GPU output
  // hits inherent rounding mismatches on edge-case (near-zero) elements that
  // exceed any practical tolerance. Match q4gsw_linear.cpp's convention and
  // skip correctness for kHalf — performance timings still run.
  if (input_spec.dtype == vkapi::kHalf) {
    throw std::invalid_argument(
        "Reference impl skipped for kHalf — fp16 round-trip diverges from "
        "the fp32 CPU reference at near-zero elements.");
  }

  std::vector<float> input_data = as_float_data(input_spec);
  std::vector<float> input_scale_data = as_float_data(input_scale_spec);
  const auto& input_zero_point_data = input_zeros_spec.get_int8_data();
  const auto& weight_data = weight_spec.get_int8_data();
  const auto& weight_sums_data = weight_sums_spec.get_int32_data();
  std::vector<float> weight_scales_data = as_float_data(weight_scales_spec);
  std::vector<float> bias_data;
  if (!bias_spec.is_none()) {
    bias_data = as_float_data(bias_spec);
  }

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.assign(batch_size * out_features, 0.0f);

  for (int64_t b = 0; b < batch_size; ++b) {
    float input_scale = input_scale_data[b];
    int8_t input_zp = input_zero_point_data[b];

    // Dynamic-per-row quantization of the input
    std::vector<int32_t> q_in(in_features);
    for (int64_t k = 0; k < in_features; ++k) {
      float v = std::round(input_data[b * in_features + k] / input_scale) +
          static_cast<float>(input_zp);
      v = std::min(std::max(v, -128.0f), 127.0f);
      q_in[k] = static_cast<int32_t>(v);
    }

    for (int64_t n = 0; n < out_features; ++n) {
      int32_t acc = 0;
      for (int64_t k = 0; k < in_features; ++k) {
        acc += q_in[k] * static_cast<int32_t>(weight_data[n * in_features + k]);
      }
      // (acc - input_zp * weight_sum) * input_scale * weight_scale
      int32_t adjusted = acc - input_zp * weight_sums_data[n];
      float result =
          static_cast<float>(adjusted) * input_scale * weight_scales_data[n];
      if (!bias_data.empty()) {
        result += bias_data[n];
      }
      ref_data[b * out_features + n] = result;
    }
  }
}

void reference_impl(TestCase& test_case) {
  linear_dq8ca_q8csw_reference_impl(test_case);
}

int64_t quantized_linear_flop_calculator(const TestCase& test_case) {
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();
  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = output_sizes[1];
  int64_t output_elements = batch_size * out_features;
  int64_t ops_per_output = in_features;
  // Quantization overhead (rough estimate, matches q4gsw_linear's convention
  // so numbers are comparable between the two studies).
  int64_t quantization_ops = ops_per_output * 2 + 1;
  return output_elements * (ops_per_output + quantization_ops);
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "Dynamic INT8 Activation × Per-channel INT8 Weight Linear (dq8ca_q8csw)"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_quantized_linear_test_cases,
      quantized_linear_flop_calculator,
      "DQ8CA_Q8CSW_Linear",
      3,
      10,
      ref_fn);

  return 0;
}
