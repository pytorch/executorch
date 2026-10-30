// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/pixel_shuffle/pixel_shuffle.h>

#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_pixel_shuffle {

namespace {

// All (in_layout, out_layout) pairs across the channels-packed int8 family.
// CONV2D is a Python-level alias that resolves to 4C1W at the C++ runtime, so
// it is not listed separately -- it would just re-run the 4C1W kernel path.
const std::vector<std::pair<std::string, std::string>> kLayoutPairs = {
    {"4W4C", "4W4C"},
    {"4W4C", "4C1W"},
    {"4C1W", "4W4C"},
    {"4C1W", "4C1W"},
};

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& in_shapes,
    int upscale_factor,
    const std::vector<bool>& same_qparams,
    const std::string& test_case_name) {
  std::vector<TestCase> test_cases;

  for (const auto& shape : in_shapes) {
    for (bool same_qp : same_qparams) {
      for (const auto& layouts : kLayoutPairs) {
        PixelShuffleConfig cfg;
        cfg.in_shape = shape;
        cfg.upscale_factor = upscale_factor;
        cfg.same_qparams = same_qp;
        cfg.in_layout = layouts.first;
        cfg.out_layout = layouts.second;
        cfg.test_case_name = test_case_name;

        test_cases.push_back(create_test_case_from_config(cfg));
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
