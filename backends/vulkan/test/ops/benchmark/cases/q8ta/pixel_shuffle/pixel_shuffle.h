// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_pixel_shuffle {

constexpr int64_t kRefDimSizeLimit = 512;

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

TestCase create_test_case_from_config(const PixelShuffleConfig& config);

void q8ta_pixel_shuffle_reference_impl(TestCase& test_case);

} // namespace q8ta_pixel_shuffle
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
