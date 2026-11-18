// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace vkcompute::shader_profiling::reference {
namespace {

double at(const Tensor& t, int64_t n, int64_t c, int64_t y, int64_t x) {
  const std::vector<int64_t>& s = t.sizes;
  return t.values[((n * s[1] + c) * s[2] + y) * s[3] + x];
}

// Bilinear sample at pixel coordinate (px, py) with border padding.
double
sample_border(const Tensor& t, int64_t n, int64_t c, double px, double py) {
  const int64_t h = t.sizes[2], w = t.sizes[3];
  px = std::clamp(px, 0.0, double(w - 1));
  py = std::clamp(py, 0.0, double(h - 1));
  const int64_t x0 = int64_t(std::floor(px)), y0 = int64_t(std::floor(py));
  const int64_t x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const double wx = px - double(x0), wy = py - double(y0);
  const double top = at(t, n, c, y0, x0) * (1 - wx) + at(t, n, c, y0, x1) * wx;
  const double bottom =
      at(t, n, c, y1, x0) * (1 - wx) + at(t, n, c, y1, x1) * wx;
  return top * (1 - wy) + bottom * wy;
}

// F.interpolate(mode="bilinear", align_corners=False) of t to out_h x out_w,
// evaluated at output pixel (x, y).
double resample(
    const Tensor& t,
    int64_t n,
    int64_t c,
    int64_t x,
    int64_t y,
    int64_t out_w,
    int64_t out_h) {
  const double sx = std::max(
      (double(x) + 0.5) * double(t.sizes[3]) / double(out_w) - 0.5, 0.0);
  const double sy = std::max(
      (double(y) + 0.5) * double(t.sizes[2]) / double(out_h) - 0.5, 0.0);
  return sample_border(t, n, c, sx, sy);
}

Tensor
warp_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& flow = tensor_arg(spec, 1, tensors);
  require(
      input.sizes.size() == 4 && flow.sizes.size() == 4 &&
          flow.sizes[0] == input.sizes[0] && flow.sizes[1] == 2 &&
          flow.sizes[2] == input.sizes[2] && flow.sizes[3] == input.sizes[3],
      "warp requires input [N, C, H, W] and flow [N, 2, H, W]");
  const int64_t c = input.sizes[1], h = input.sizes[2], w = input.sizes[3];
  Tensor out{input.sizes, std::vector<double>(input.values.size())};
  parallel_for(out.values.size(), threads, [&](size_t i) {
    const int64_t x = int64_t(i) % w, y = int64_t(i) / w % h;
    const int64_t ch = int64_t(i) / (w * h) % c, n = int64_t(i) / (w * h * c);
    out.values[i] = sample_border(
        input,
        n,
        ch,
        double(x) + at(flow, n, 0, y, x),
        double(y) + at(flow, n, 1, y, x));
  });
  return out;
}

Tensor warp_blend_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& img0 = tensor_arg(spec, 0, tensors);
  const Tensor& img1 = tensor_arg(spec, 1, tensors);
  const Tensor& flow = tensor_arg(spec, 2, tensors);
  const Tensor& mask = tensor_arg(spec, 3, tensors);
  require(
      img0.sizes.size() == 4 && img1.sizes == img0.sizes &&
          flow.sizes.size() == 4 && mask.sizes.size() == 4 &&
          flow.sizes[0] == img0.sizes[0] && flow.sizes[1] == 4 &&
          mask.sizes[0] == img0.sizes[0] && mask.sizes[1] == 1 &&
          mask.sizes[2] == flow.sizes[2] && mask.sizes[3] == flow.sizes[3],
      "warp_blend requires img0/img1 [N, C, H, W], flow [N, 4, h, w] and "
      "mask [N, 1, h, w]");
  const int64_t c = img0.sizes[1], h = img0.sizes[2], w = img0.sizes[3];
  // Flow displacements are in flow pixels; scale them to output pixels.
  const double sx = double(w) / double(flow.sizes[3]);
  const double sy = double(h) / double(flow.sizes[2]);
  Tensor out{img0.sizes, std::vector<double>(img0.values.size())};
  parallel_for(out.values.size(), threads, [&](size_t i) {
    const int64_t x = int64_t(i) % w, y = int64_t(i) / w % h;
    const int64_t ch = int64_t(i) / (w * h) % c, n = int64_t(i) / (w * h * c);
    std::array<double, 4> f{};
    for (int64_t k = 0; k < 4; ++k) {
      f[k] = resample(flow, n, k, x, y, w, h);
    }
    const double m = 1.0 / (1.0 + std::exp(-resample(mask, n, 0, x, y, w, h)));
    const double warped0 = sample_border(
        img0, n, ch, double(x) + f[0] * sx, double(y) + f[1] * sy);
    const double warped1 = sample_border(
        img1, n, ch, double(x) + f[2] * sx, double(y) + f[3] * sy);
    out.values[i] = std::clamp(warped0 * m + warped1 * (1 - m), 0.0, 1.0);
  });
  return out;
}

} // namespace

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(warp, warp_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.warp.default, warp);
  REGISTER_REFERENCE_FN(warp_blend, warp_blend_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.warp_blend.default, warp_blend);
}
} // namespace vkcompute::shader_profiling::reference
