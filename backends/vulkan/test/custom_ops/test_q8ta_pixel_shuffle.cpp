// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "utils.h"

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 512;

// Test op signature in TestQ8taPixelShuffle.cpp:
//   test_q8ta_pixel_shuffle(fp_in, in_scale, in_zp, out_scale, out_zp,
//                           upscale_factor, in_layout, out_layout) -> fp_out
// Implementation: fused fast-path kernel. The in_layout / out_layout strings
// select the channels-packed int8x4 layout used for the temporary quantized
// tensors. Supported values: "4W4C", "4C1W".
// (PACKED_INT8_CONV2D is a Python/serialization-level alias that the runtime
// resolves to kPackedInt8_4C1W, so it is not exercised separately here -- it
// would only re-test the same C++ kernel path as "4C1W".)

struct PixelShuffleConfig {
  std::vector<int64_t> in_shape; // [N, C*r*r, H, W]
  int upscale_factor;
  bool same_qparams; // if true, in_scale == out_scale and in_zp == out_zp
  std::string in_layout = "4W4C";
  std::string out_layout = "4W4C";
  std::string test_case_name = "ACCU";
  std::string op_name = "test_q8ta_pixel_shuffle";
};

TestCase create_test_case_from_config(const PixelShuffleConfig& config) {
  TestCase test_case;

  std::string shape_str = shape_string(config.in_shape);
  std::string qp_label = config.same_qparams ? "same_qp" : "diff_qp";
  std::string layout_label =
      "[" + config.in_layout + "->" + config.out_layout + "]";
  std::string test_name = config.test_case_name + "  In=" + shape_str +
      "  r=" + std::to_string(config.upscale_factor) + "  " + qp_label + "  " +
      layout_label;
  test_case.set_name(test_name);

  std::string operator_name = "test_etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // FP input: shape [N, C_in, H, W]
  ValueSpec input_tensor(
      config.in_shape,
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  float input_scale_val = 0.007112f;
  ValueSpec input_scale(input_scale_val);
  int32_t input_zp_val = 0;
  ValueSpec input_zp(input_zp_val);

  float output_scale_val = config.same_qparams ? input_scale_val : 0.013f;
  ValueSpec output_scale(output_scale_val);
  int32_t output_zp_val = config.same_qparams ? input_zp_val : 5;
  ValueSpec output_zp(output_zp_val);

  ValueSpec upscale_factor(static_cast<int32_t>(config.upscale_factor));

  ValueSpec in_layout_spec = ValueSpec::make_string(config.in_layout);
  ValueSpec out_layout_spec = ValueSpec::make_string(config.out_layout);

  // Output shape
  std::vector<int64_t> out_shape = {
      config.in_shape[0],
      config.in_shape[1] / (config.upscale_factor * config.upscale_factor),
      config.in_shape[2] * config.upscale_factor,
      config.in_shape[3] * config.upscale_factor};
  ValueSpec output_tensor(
      out_shape,
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zp);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zp);
  test_case.add_input_spec(upscale_factor);
  test_case.add_input_spec(in_layout_spec);
  test_case.add_input_spec(out_layout_spec);
  test_case.add_output_spec(output_tensor);

  // Tolerance: ~1 quant step in the bigger of the two scales.
  float tol = std::max(input_scale_val, output_scale_val) + 1e-4f;
  test_case.set_abs_tolerance(tol);

  // Filter shaders that are just measurement overhead (staging copies and the
  // surrounding quantize/dequantize that wrap the operation under test).
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// All (in_layout, out_layout) pairs across the channels-packed int8 family.
// CONV2D is a Python-level alias that resolves to 4C1W at the C++ runtime, so
// it is not listed separately -- it would just re-run the 4C1W kernel path.
static const std::vector<std::pair<std::string, std::string>>&
get_layout_pairs() {
  static const std::vector<std::pair<std::string, std::string>> layout_pairs = {
      {"4W4C", "4W4C"},
      {"4W4C", "4C1W"},
      {"4C1W", "4W4C"},
      {"4C1W", "4C1W"},
  };
  return layout_pairs;
}

std::vector<TestCase> generate_correctness_cases() {
  std::vector<TestCase> test_cases;

  // Small shapes, all use r=2 (the only factor needed by the model).
  // Shape format is the *input* shape [N, C_in, H, W] where C_in = C_out * r*r.
  std::vector<std::vector<int64_t>> shapes = {
      // Small even W to be a multiple of 4 after upscaling.
      {1, 16, 4, 4}, // out: [1, 4, 8, 8]
      {1, 24, 8, 4}, // out: [1, 6, 16, 8]
      {1, 32, 12, 8}, // out: [1, 8, 24, 16]
      {1, 96, 16, 9}, // out: [1, 24, 32, 18] - first model shape
  };

  for (const auto& shape : shapes) {
    for (bool same_qp : {true, false}) {
      for (const auto& layouts : get_layout_pairs()) {
        PixelShuffleConfig cfg;
        cfg.in_shape = shape;
        cfg.upscale_factor = 2;
        cfg.same_qparams = same_qp;
        cfg.in_layout = layouts.first;
        cfg.out_layout = layouts.second;
        cfg.test_case_name = "ACCU";

        test_cases.push_back(create_test_case_from_config(cfg));
      }
    }
  }

  return test_cases;
}

std::vector<TestCase> generate_perf_cases() {
  std::vector<TestCase> test_cases;

  // Model perf shapes (output shapes from the RefineNet decoder; we compute
  // the input shape as [N, C_out * r*r, H_out / r, W_out / r]).
  // Output shapes: [1, 24, 32, 18], [1, 24, 64, 36], [1, 24, 128, 72],
  // [1, 24, 256, 144]. For r=2, in shapes = [1, 96, 16, 9], etc.
  std::vector<std::vector<int64_t>> in_shapes = {
      {1, 96, 16, 9},
      {1, 96, 32, 18},
      {1, 96, 64, 36},
      {1, 96, 128, 72},
  };

  for (const auto& shape : in_shapes) {
    for (const auto& layouts : get_layout_pairs()) {
      PixelShuffleConfig cfg;
      cfg.in_shape = shape;
      cfg.upscale_factor = 2;
      cfg.same_qparams = true; // residual-style: scales match
      cfg.in_layout = layouts.first;
      cfg.out_layout = layouts.second;
      cfg.test_case_name = "PERF";

      test_cases.push_back(create_test_case_from_config(cfg));
    }
  }

  return test_cases;
}

std::vector<TestCase> generate_all_cases() {
  std::vector<TestCase> all = generate_correctness_cases();
  std::vector<TestCase> perf = generate_perf_cases();
  for (auto& tc : perf) {
    all.push_back(tc);
  }
  return all;
}

// Reference: quantize input, do PyTorch-equivalent pixel shuffle, requantize,
// then dequantize for comparison.
void q8ta_pixel_shuffle_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& upscale_factor_spec = test_case.inputs()[idx++];

  ValueSpec& output_spec = test_case.outputs()[0];

  const auto in_sizes = input_spec.get_tensor_sizes();
  for (auto d : in_sizes) {
    if (d > kRefDimSizeLimit) {
      throw std::invalid_argument("Dim exceeds reference compute limit");
    }
  }

  const int64_t N = in_sizes[0];
  const int64_t C_in = in_sizes[1];
  const int64_t H_in = in_sizes[2];
  const int64_t W_in = in_sizes[3];
  const int32_t r = upscale_factor_spec.get_int_value();
  const int64_t C_out = C_in / (r * r);
  const int64_t H_out = H_in * r;
  const int64_t W_out = W_in * r;

  const float input_scale = input_scale_spec.get_float_value();
  const int32_t input_zp = input_zp_spec.get_int_value();
  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zp = output_zp_spec.get_int_value();
  const int32_t qmin = -128;
  const int32_t qmax = 127;

  const auto& input_data = input_spec.get_float_data();
  auto& ref = output_spec.get_ref_float_data();
  ref.resize(N * C_out * H_out * W_out);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c_out = 0; c_out < C_out; ++c_out) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          const int64_t c_in = c_out * r * r + (oh % r) * r + (ow % r);
          const int64_t ih = oh / r;
          const int64_t iw = ow / r;
          const int64_t in_idx = ((n * C_in + c_in) * H_in + ih) * W_in + iw;
          const float fp_in = input_data[in_idx];

          // Quantize with input qparams
          float qf = std::round(fp_in / input_scale) + input_zp;
          qf = std::max(qf, static_cast<float>(qmin));
          qf = std::min(qf, static_cast<float>(qmax));
          int32_t q_in = static_cast<int32_t>(qf);

          // Dequantize back to fp using the input qparams (this models the
          // dequantize node in the chain)
          float dq = (q_in - input_zp) * input_scale;

          // Requantize to int8 with output qparams
          float rqf = std::round(dq / output_scale) + output_zp;
          rqf = std::max(rqf, static_cast<float>(qmin));
          rqf = std::min(rqf, static_cast<float>(qmax));
          int32_t q_out = static_cast<int32_t>(rqf);

          // Final dequantize to fp for comparison
          float fp_out = (q_out - output_zp) * output_scale;

          const int64_t out_idx =
              ((n * C_out + c_out) * H_out + oh) * W_out + ow;
          ref[out_idx] = fp_out;
        }
      }
    }
  }
}

int main(int /*argc*/, char* /*argv*/[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Q8TA PixelShuffle Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = q8ta_pixel_shuffle_reference_impl;

  auto results = execute_test_cases(
      generate_all_cases,
      "Q8taPixelShuffle",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      ref_fn);

  return 0;
}
