// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {

// Component structs for better readability
struct KernelSize {
  int32_t h;
  int32_t w;

  KernelSize(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Stride {
  int32_t h;
  int32_t w;

  Stride(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Padding {
  int32_t h;
  int32_t w;

  Padding(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Dilation {
  int32_t h;
  int32_t w;

  Dilation(int32_t height = 1, int32_t width = 1) : h(height), w(width) {}
};

struct OutInChannels {
  int32_t out;
  int32_t in;

  OutInChannels(int32_t out_channels, int32_t in_channels)
      : out(out_channels), in(in_channels) {}
};

struct InputSize2D {
  int32_t h;
  int32_t w;

  InputSize2D(int32_t height, int32_t width) : h(height), w(width) {}
};

// Conv2d configuration struct
struct Conv2dConfig {
  OutInChannels channels;
  InputSize2D input_size;
  KernelSize kernel;
  Stride stride;
  Padding padding;
  Dilation dilation;
  int32_t groups; // Number of groups for grouped convolution
  std::string test_case_name = "placeholder";
  std::string op_name = "conv2d";

  // Calculate output dimensions
  int64_t get_output_height() const {
    return (input_size.h + 2 * padding.h - dilation.h * (kernel.h - 1) - 1) /
        stride.h +
        1;
  }

  int64_t get_output_width() const {
    return (input_size.w + 2 * padding.w - dilation.w * (kernel.w - 1) - 1) /
        stride.w +
        1;
  }
};

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
