// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/pixel_shuffle/pixel_shuffle.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_pixel_shuffle {

namespace {

// Channels-packed int8 layouts. CONV2D is a Python-level alias that resolves to
// 4C or 4C1W at the C++ runtime, so it is not listed separately.
const std::vector<std::string> kLayouts = {"4W4C", "4C", "4C1W"};

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& in_shapes,
    int upscale_factor,
    const std::vector<bool>& same_qparams,
    const std::string& test_case_name) {
  std::vector<TestCase> test_cases;

  for (const auto& shape : in_shapes) {
    for (bool same_qp : same_qparams) {
      for (const std::string& in_layout : kLayouts) {
        for (const std::string& out_layout : kLayouts) {
          PixelShuffleConfig cfg;
          cfg.in_shape = shape;
          cfg.upscale_factor = upscale_factor;
          cfg.same_qparams = same_qp;
          cfg.in_layout = in_layout;
          cfg.out_layout = out_layout;
          cfg.test_case_name = test_case_name;

          test_cases.push_back(create_test_case_from_config(cfg));
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/pixel_shuffle", "correctness") {
  // Small shapes, all use r=2 (the only factor needed by the model).
  // Shape format is the *input* shape [N, C_in, H, W] where C_in = C_out * r*r.
  const std::vector<std::vector<int64_t>> in_shapes = {
      // Small even W to be a multiple of 4 after upscaling.
      {1, 16, 4, 4}, // out: [1, 4, 8, 8]
      {1, 24, 8, 4}, // out: [1, 6, 16, 8]
      {1, 32, 12, 8}, // out: [1, 8, 24, 16]
      {1, 96, 16, 9}, // out: [1, 24, 32, 18] - first model shape
      {1, 4, 5, 7}, // out: [1, 1, 10, 14] - 3 padding channel lanes
      {2, 32, 6, 5}, // out: [2, 8, 12, 10] - batch > 1
  };
  const std::vector<bool> same_qparams = {true, false};
  return {
      generate_test_cases(
          in_shapes, /*upscale_factor=*/2, same_qparams, "ACCU"),
      q8ta_pixel_shuffle_reference_impl};
}

REGISTER_TEST_CASE_SET("q8ta/pixel_shuffle", "performance") {
  // Model perf shapes (output shapes from the RefineNet decoder; we compute
  // the input shape as [N, C_out * r*r, H_out / r, W_out / r]).
  // Output shapes: [1, 24, 32, 18], [1, 24, 64, 36], [1, 24, 128, 72],
  // [1, 24, 256, 144]. For r=2, in shapes = [1, 96, 16, 9], etc.
  const std::vector<std::vector<int64_t>> in_shapes = {
      {1, 96, 16, 9},
      {1, 96, 32, 18},
      {1, 96, 64, 36},
      {1, 96, 128, 72},
      // Upsampling decoder with 64 output channels.
      {1, 256, 16, 16},
      {1, 256, 32, 32},
      {1, 256, 64, 64},
      {1, 256, 128, 128},
      {1, 256, 256, 256},
  };
  const std::vector<bool> same_qparams = {true}; // residual-style: scales match
  return {
      generate_test_cases(
          in_shapes, /*upscale_factor=*/2, same_qparams, "PERF"),
      q8ta_pixel_shuffle_reference_impl};
}

} // namespace q8ta_pixel_shuffle
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
