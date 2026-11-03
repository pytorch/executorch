/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <algorithm>
#include <cmath>
#include <type_traits>

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ArrayRef;
using executorch::aten::OptionalArrayRef;
using std::optional;

namespace {

// Keys' cubic with a = -0.5, the one ATen uses when it antialiases.
template <typename T>
T cubic_aa_filter(T x) {
  constexpr T a = -0.5;
  x = std::abs(x);
  if (x < 1) {
    return ((a + 2) * x - (a + 3)) * x * x + 1;
  }
  if (x < 2) {
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  }
  return 0;
}

// The inputs that each output reads along one dim: `size[o]` of them from
// `first[o]` on, with their weights at `values + o * max_size`.
template <typename T>
struct Weights {
  int64_t max_size;
  int64_t* first;
  int64_t* size;
  T* values;
};

// The same in fixed point with `precision` fraction bits, as ATen's uint8 path
// uses them.
struct FixedPointWeights {
  int64_t max_size;
  int64_t* first;
  int64_t* size;
  int16_t* values;
  int precision;
};

// Returns null when the memory is not available, including when its size does
// not fit in size_t.
template <typename U>
U* allocate_temp(KernelRuntimeContext& ctx, int64_t count) {
  if (static_cast<uint64_t>(count) > SIZE_MAX / sizeof(U)) {
    return nullptr;
  }
  Result<void*> memory =
      ctx.allocate_temp(static_cast<size_t>(count) * sizeof(U), alignof(U));
  return memory.ok() ? static_cast<U*>(memory.get()) : nullptr;
}

// What the windows of one dim share: the shrink ratio, and the filter support
// widened by it.
template <typename T>
struct Filter {
  int64_t in_size;
  int64_t max_size;
  T ratio;
  T support;
  T invscale;
};

template <typename T>
Error make_filter(
    int64_t in_size,
    int64_t out_size,
    bool align_corners,
    const optional<double> scale,
    Filter<T>& filter) {
  const T ratio =
      area_pixel_compute_scale<T>(in_size, out_size, align_corners, scale);
  // F.interpolate never makes the ratio larger than the input size, and the
  // + 1 leaves room for rounding. A larger ratio comes from a scale that is
  // too small for the input, and could overflow the window bounds below.
  ET_CHECK_OR_RETURN_ERROR(
      ratio <= in_size + 1,
      InvalidArgument,
      "_upsample_bicubic2d_aa got a scale that is too small for its input "
      "size");
  filter.in_size = in_size;
  filter.ratio = ratio;
  // The filter spans 4 inputs, widened by the shrink factor.
  filter.support = static_cast<T>((ratio >= 1.0) ? (4 * 0.5) * ratio : 4 * 0.5);
  filter.invscale = static_cast<T>((ratio >= 1.0) ? 1.0 / ratio : 1.0);
  filter.max_size = static_cast<int64_t>(std::ceil(filter.support)) * 2 + 1;
  return Error::Ok;
}

// Computes the window of output `o` and its weights as ATen's antialiased CPU
// kernel does, expression for expression, so that they round the same way.
template <typename T>
void compute_window(
    const Filter<T>& filter,
    int64_t o,
    int64_t& first,
    int64_t& size,
    T* values) {
  const T center = static_cast<T>(filter.ratio * (o + 0.5));
  const int64_t start =
      std::max(static_cast<int64_t>(center - filter.support + 0.5), int64_t(0));
  size = std::clamp(
      std::min(
          static_cast<int64_t>(center + filter.support + 0.5), filter.in_size) -
          start,
      int64_t(0),
      filter.max_size);
  T total = 0;
  for (const auto j : c10::irange(size)) {
    values[j] = cubic_aa_filter<T>(
        static_cast<T>((j + start - center + 0.5) * filter.invscale));
    total += values[j];
  }
  if (total != 0) {
    for (const auto j : c10::irange(size)) {
      values[j] /= total;
    }
  }
  // An empty window reads nothing, but its start still has to point into the
  // input.
  first = std::min(start, filter.in_size - 1);
}

template <typename T>
Error compute_weights(
    KernelRuntimeContext& ctx,
    int64_t in_size,
    int64_t out_size,
    bool align_corners,
    const optional<double> scale,
    Weights<T>& weights) {
  Filter<T> filter{};
  ET_CHECK_OK_OR_RETURN_ERROR(
      make_filter(in_size, out_size, align_corners, scale, filter));
  weights.max_size = filter.max_size;
  weights.first = allocate_temp<int64_t>(ctx, out_size);
  weights.size = weights.first != nullptr
      ? allocate_temp<int64_t>(ctx, out_size)
      : nullptr;
  weights.values = weights.size != nullptr
      ? allocate_temp<T>(ctx, out_size * weights.max_size)
      : nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      weights.values != nullptr,
      MemoryAllocationFailed,
      "_upsample_bicubic2d_aa could not get temporary memory for its "
      "weights");
  for (const auto o : c10::irange(out_size)) {
    compute_window(
        filter,
        o,
        weights.first[o],
        weights.size[o],
        weights.values + o * weights.max_size);
  }
  return Error::Ok;
}

// Converts the weights as ATen's uint8 path does, with the most fraction bits,
// up to 22, that keep the largest weight below 2^15. Each window is computed
// in double twice, to find the largest weight and then to convert it, so that
// only one window of doubles is kept.
Error compute_weights(
    KernelRuntimeContext& ctx,
    int64_t in_size,
    int64_t out_size,
    bool align_corners,
    const optional<double> scale,
    FixedPointWeights& weights) {
  Filter<double> filter{};
  ET_CHECK_OK_OR_RETURN_ERROR(
      make_filter(in_size, out_size, align_corners, scale, filter));
  weights.max_size = filter.max_size;
  double* exact = allocate_temp<double>(ctx, filter.max_size);
  weights.first =
      exact != nullptr ? allocate_temp<int64_t>(ctx, out_size) : nullptr;
  weights.size = weights.first != nullptr
      ? allocate_temp<int64_t>(ctx, out_size)
      : nullptr;
  weights.values = weights.size != nullptr
      ? allocate_temp<int16_t>(ctx, out_size * weights.max_size)
      : nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      weights.values != nullptr,
      MemoryAllocationFailed,
      "_upsample_bicubic2d_aa could not get temporary memory for its "
      "weights");
  double max_weight = 0;
  for (const auto o : c10::irange(out_size)) {
    compute_window(filter, o, weights.first[o], weights.size[o], exact);
    for (const auto j : c10::irange(weights.size[o])) {
      max_weight = std::max(max_weight, exact[j]);
    }
  }
  int precision = 0;
  while (precision < 22 &&
         0.5 + max_weight * (1 << (precision + 1)) < (1 << 15)) {
    ++precision;
  }
  // Only a scale that F.interpolate never gives, with windows that nearly
  // cancel out, gets here. ATen's generic CPU path fails on it too.
  ET_CHECK_OR_RETURN_ERROR(
      precision > 0,
      InvalidArgument,
      "_upsample_bicubic2d_aa got a scale whose uint8 weights do not fit in "
      "fixed point");
  for (const auto o : c10::irange(out_size)) {
    compute_window(filter, o, weights.first[o], weights.size[o], exact);
    int16_t* values = weights.values + o * weights.max_size;
    for (const auto j : c10::irange(weights.size[o])) {
      const double value = exact[j] * (1 << precision);
      // The precision only bounds the largest weight, so with such a scale a
      // negative one can still be out of range.
      ET_CHECK_OR_RETURN_ERROR(
          value > -32768.5 && value < 32767.5,
          InvalidArgument,
          "_upsample_bicubic2d_aa got a scale whose uint8 weights do not fit "
          "in fixed point");
      values[j] = static_cast<int16_t>(
          value < 0 ? static_cast<int>(-0.5 + value)
                    : static_cast<int>(0.5 + value));
    }
  }
  weights.precision = precision;
  return Error::Ok;
}

// The value of output `o`, from the inputs it reads. They start at `in` and
// are `step` elements apart.
template <typename T, typename CTYPE_IN>
T weighted_sum(
    const CTYPE_IN* in,
    int64_t step,
    const Weights<T>& weights,
    int64_t o) {
  const T* values = weights.values + o * weights.max_size;
  T sum = 0;
  for (const auto j : c10::irange(weights.size[o])) {
    sum += static_cast<T>(in[j * step]) * values[j];
  }
  return sum;
}

// ATen's uint8 sum: starts at one half so that dropping the fraction bits
// rounds, and clamps the result. It adds up in 64 bits, not in int as ATen
// does, because a scale F.interpolate never gives can make a window wide
// enough to overflow int.
uint8_t weighted_sum(
    const uint8_t* in,
    int64_t step,
    const FixedPointWeights& weights,
    int64_t o) {
  const int16_t* values = weights.values + o * weights.max_size;
  int64_t sum = int64_t{1} << (weights.precision - 1);
  for (const auto j : c10::irange(weights.size[o])) {
    sum += in[j * step] * values[j];
  }
  return static_cast<uint8_t>(
      std::clamp<int64_t>(sum >> weights.precision, 0, 255));
}

// Like ATen, resizes the width first and then the height, and copies a dim
// whose size does not change. Half precision is computed in float and the
// result is rounded once. uint8 follows ATen's fixed-point path and rounds
// after each pass. ATen's generic CPU path gives every row the first row's
// values when the input is contiguous, the output width is 1 and the height
// changes, because basic_loop_separable_1d_vertical in
// aten/src/ATen/native/cpu/UpSampleKernel.cpp reuses the first row's weights
// (https://github.com/pytorch/pytorch/issues/200568). This kernel resizes the
// height there too.
template <typename CTYPE>
void upsample_bicubic2d_aa_kernel_impl(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    bool align_corners,
    const optional<double> scale_h,
    const optional<double> scale_w,
    Tensor& out) {
  constexpr bool is_byte = std::is_same_v<CTYPE, uint8_t>;
  using T = std::conditional_t<std::is_same_v<CTYPE, double>, double, float>;
  using W = std::conditional_t<is_byte, FixedPointWeights, Weights<T>>;
  // uint8 keeps the rows between the passes in uint8, as ATen does.
  using Row = std::conditional_t<is_byte, uint8_t, T>;
  const int64_t in_h = in.size(2);
  const int64_t out_h = out.size(2);
  const int64_t out_w = out.size(3);
  const bool resize_h = in_h != out_h;
  const bool resize_w = in.size(3) != out_w;
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  if (!resize_h && !resize_w) {
    // Same sizes and dim order, so the same layout.
    std::copy(in_data, in_data + in.numel(), out_data);
    return;
  }

  W weights_h{};
  W weights_w{};
  Error error = Error::Ok;
  if (resize_h) {
    error =
        compute_weights(ctx, in_h, out_h, align_corners, scale_h, weights_h);
  }
  if (error == Error::Ok && resize_w) {
    error = compute_weights(
        ctx, in.size(3), out_w, align_corners, scale_w, weights_w);
  }
  if (error != Error::Ok) {
    ctx.fail(error);
    return;
  }

  // When both dims change, the height pass reads rows of the width pass. If
  // the windows are short, only the last `ring_size` rows are kept, which is
  // enough for one output row, and each is stored twice, `ring_size` rows
  // apart, so that the rows one output reads are always next to each other.
  // Otherwise all rows are kept, once each.
  const bool ring_wraps = 2 * weights_h.max_size < in_h;
  const int64_t ring_size = ring_wraps ? weights_h.max_size : in_h;
  Row* ring = nullptr;
  if (resize_h && resize_w) {
    ring = allocate_temp<Row>(ctx, (ring_wraps ? 2 : 1) * ring_size * out_w);
    ET_KERNEL_CHECK_MSG(
        ctx,
        ring != nullptr,
        MemoryAllocationFailed,
        ,
        "_upsample_bicubic2d_aa could not get temporary memory for the resized "
        "rows");
  }

  const auto in_strides = in.strides();
  const auto out_strides = out.strides();
  for (const auto n : c10::irange(out.size(0))) {
    for (const auto c : c10::irange(out.size(1))) {
      const CTYPE* in_plane = in_data + n * in_strides[0] + c * in_strides[1];
      CTYPE* out_plane = out_data + n * out_strides[0] + c * out_strides[1];
      // The width pass has already resized every input row before this one.
      int64_t next_row = 0;
      for (const auto h : c10::irange(out_h)) {
        CTYPE* out_row = out_plane + h * out_strides[2];
        if (!resize_h) {
          const CTYPE* in_row = in_plane + h * in_strides[2];
          for (const auto w : c10::irange(out_w)) {
            out_row[w * out_strides[3]] = static_cast<CTYPE>(weighted_sum(
                in_row + weights_w.first[w] * in_strides[3],
                in_strides[3],
                weights_w,
                w));
          }
          continue;
        }
        const int64_t first = weights_h.first[h];
        if (!resize_w) {
          for (const auto w : c10::irange(out_w)) {
            out_row[w * out_strides[3]] = static_cast<CTYPE>(weighted_sum(
                in_plane + first * in_strides[2] + w * in_strides[3],
                in_strides[2],
                weights_h,
                h));
          }
          continue;
        }
        const int64_t end = first + weights_h.size[h];
        for (int64_t row = std::max(next_row, first); row < end; ++row) {
          const CTYPE* in_row = in_plane + row * in_strides[2];
          Row* resized = ring + (row % ring_size) * out_w;
          for (const auto w : c10::irange(out_w)) {
            resized[w] = weighted_sum(
                in_row + weights_w.first[w] * in_strides[3],
                in_strides[3],
                weights_w,
                w);
            if (ring_wraps) {
              resized[w + ring_size * out_w] = resized[w];
            }
          }
        }
        next_row = std::max(next_row, end);
        const Row* rows = ring + (first % ring_size) * out_w;
        for (const auto w : c10::irange(out_w)) {
          out_row[w * out_strides[3]] =
              static_cast<CTYPE>(weighted_sum(rows + w, out_w, weights_h, h));
        }
      }
    }
  }
}

} // namespace

Tensor& _upsample_bicubic2d_aa_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const ArrayRef<int64_t> output_size,
    bool align_corners,
    const optional<double> scale_h,
    const optional<double> scale_w,
    Tensor& out) {
  const OptionalArrayRef<int64_t> size(output_size);
  ET_KERNEL_CHECK(
      ctx,
      check_upsample_2d_common_args(in, size, {}, out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK_MSG(
      ctx,
      in.size(2) > 0 && in.size(3) > 0,
      InvalidArgument,
      out,
      "_upsample_bicubic2d_aa needs a non-empty input height and width");

  double unused_scale_h, unused_scale_w;
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_upsample_2d(in, size, {}, unused_scale_h, unused_scale_w, out) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor");

  ET_SWITCH_FLOATHBF16_TYPES_AND(
      Byte, in.scalar_type(), ctx, "_upsample_bicubic2d_aa.out", CTYPE, [&]() {
        upsample_bicubic2d_aa_kernel_impl<CTYPE>(
            ctx, in, align_corners, scale_h, scale_w, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
