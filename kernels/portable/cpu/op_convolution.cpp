/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>

#include <algorithm>

#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;
using SizesArrayRef = executorch::aten::ArrayRef<executorch::aten::SizesType>;
using StridesArrayRef =
    executorch::aten::ArrayRef<executorch::aten::StridesType>;

namespace {

struct IndexRange {
  int64_t begin;
  int64_t end;
};

// The range helpers do not depend on the dtype, so out of line one copy serves
// every dtype and spatial dim.
ET_NOINLINE IndexRange transposed_kernel_range(
    int64_t begin,
    int64_t end,
    int64_t input_size,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
  return {
      std::max(
          int64_t{0},
          (begin + padding - (input_size - 1) * stride + dilation - 1) /
              dilation),
      std::min(kernel_size, (end - 1 + padding) / dilation + 1)};
}

ET_NOINLINE IndexRange transposed_input_range(
    int64_t begin,
    int64_t end,
    int64_t input_size,
    int64_t offset,
    int64_t stride) {
  return {
      std::max(int64_t{0}, (begin - offset + stride - 1) / stride),
      std::min(input_size, (end - 1 - offset) / stride + 1)};
}

// Keep the tile buffer out of the regular convolution's stack frame.
template <typename CTYPE, typename LoadFn = CTYPE (*)(const void*)>
ET_NOINLINE void transposed_conv3d_impl(
    const CTYPE* const in_ptr,
    SizesArrayRef in_sizes,
    StridesArrayRef in_strides,
    const CTYPE* const w_ptr,
    SizesArrayRef w_sizes,
    StridesArrayRef w_strides,
    const std::optional<Tensor>& bias,
    const char* const bias_ptr,
    LoadFn load_bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    CTYPE* const out_ptr,
    SizesArrayRef out_sizes,
    StridesArrayRef out_strides,
    const size_t batch,
    const size_t group,
    const size_t out_c) {
  size_t in_C = in_sizes[1];
  size_t out_C = out_sizes[1];

  int64_t out_D = out_sizes[2];
  int64_t in_D = in_sizes[2];
  int64_t w_D = w_sizes[2];

  int64_t out_H = out_sizes[3];
  int64_t in_H = in_sizes[3];
  int64_t w_H = w_sizes[3];

  int64_t out_W = out_sizes[4];
  int64_t in_W = in_sizes[4];
  int64_t w_W = w_sizes[4];

  size_t in_C_per_group = in_C / groups;
  size_t in_c_start = group * in_C_per_group;

  size_t out_C_per_group = out_C / groups;
  size_t out_c_start = group * out_C_per_group;

  const int64_t stride_z = stride[0];
  const int64_t padding_z = padding[0];
  const int64_t dilation_z = dilation[0];
  const int64_t stride_y = stride[1];
  const int64_t padding_y = padding[1];
  const int64_t dilation_y = dilation[1];
  const int64_t stride_x = stride[2];
  const int64_t padding_x = padding[2];
  const int64_t dilation_x = dilation[2];

  using COMPUTE_T =
      typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
  // Reuse weights across a bounded tile, narrowing only after reduction.
  constexpr int64_t kTileHeight = 8;
  constexpr int64_t kTileWidth = 32;
  for (int64_t out_z = 0; out_z < out_D; ++out_z) {
    const auto kernel_z = transposed_kernel_range(
        out_z, out_z + 1, in_D, w_D, stride_z, padding_z, dilation_z);
    for (int64_t tile_y = 0; tile_y < out_H; tile_y += kTileHeight) {
      const int64_t tile_y_end = std::min(out_H, tile_y + kTileHeight);
      const auto kernel_y = transposed_kernel_range(
          tile_y, tile_y_end, in_H, w_H, stride_y, padding_y, dilation_y);
      for (int64_t tile_x = 0; tile_x < out_W; tile_x += kTileWidth) {
        const int64_t tile_x_end = std::min(out_W, tile_x + kTileWidth);
        const auto kernel_x = transposed_kernel_range(
            tile_x, tile_x_end, in_W, w_W, stride_x, padding_x, dilation_x);
        COMPUTE_T accum[kTileHeight * kTileWidth] = {};
        for (int64_t w_z = kernel_z.begin; w_z < kernel_z.end; ++w_z) {
          // A tile is one output deep, so at most one input depth lands on it.
          const auto input_z = transposed_input_range(
              out_z, out_z + 1, in_D, w_z * dilation_z - padding_z, stride_z);
          if (input_z.begin >= input_z.end) {
            continue;
          }
          const size_t in_z_idx =
              batch * in_strides[0] + input_z.begin * in_strides[2];
          const size_t w_z_idx =
              (out_c - out_c_start) * w_strides[1] + w_z * w_strides[2];
          for (int64_t w_y = kernel_y.begin; w_y < kernel_y.end; ++w_y) {
            const int64_t offset_y = w_y * dilation_y - padding_y;
            const auto input_y = transposed_input_range(
                tile_y, tile_y_end, in_H, offset_y, stride_y);
            if (input_y.begin >= input_y.end) {
              continue;
            }
            for (int64_t w_x = kernel_x.begin; w_x < kernel_x.end; ++w_x) {
              const int64_t offset_x = w_x * dilation_x - padding_x;
              const auto input_x = transposed_input_range(
                  tile_x, tile_x_end, in_W, offset_x, stride_x);
              if (input_x.begin >= input_x.end) {
                continue;
              }
              for (const auto in_c :
                   c10::irange(in_c_start, in_c_start + in_C_per_group)) {
                const size_t w_idx = w_z_idx + in_c * w_strides[0] +
                    w_y * w_strides[3] + w_x * w_strides[4];
                const COMPUTE_T w_val = w_ptr[w_idx];
                int64_t out_y = input_y.begin * stride_y + offset_y - tile_y;
                for (int64_t in_y = input_y.begin; in_y < input_y.end; ++in_y) {
                  size_t in_idx = in_z_idx + in_c * in_strides[1] +
                      in_y * in_strides[3] + input_x.begin * in_strides[4];
                  int64_t out_x = input_x.begin * stride_x + offset_x - tile_x;
                  for (int64_t in_x = input_x.begin; in_x < input_x.end;
                       ++in_x) {
                    const COMPUTE_T in_val = in_ptr[in_idx];
                    accum[out_y * kTileWidth + out_x] += in_val * w_val;
                    in_idx += in_strides[4];
                    out_x += stride_x;
                  }
                  out_y += stride_y;
                }
              }
            }
          }
        }
        const COMPUTE_T bias_val = bias_ptr == nullptr
            ? COMPUTE_T{0}
            : load_bias(&bias_ptr[out_c * bias.value().element_size()]);
        for (int64_t out_y = tile_y; out_y < tile_y_end; ++out_y) {
          size_t out_idx = batch * out_strides[0] + out_c * out_strides[1] +
              out_z * out_strides[2] + out_y * out_strides[3] +
              tile_x * out_strides[4];
          for (int64_t out_x = tile_x; out_x < tile_x_end; ++out_x) {
            out_ptr[out_idx] =
                accum[(out_y - tile_y) * kTileWidth + out_x - tile_x] +
                bias_val;
            out_idx += out_strides[4];
          }
        }
      }
    }
  }
}

template <typename CTYPE, typename LoadFn = CTYPE (*)(const void*)>
void conv3d_impl(
    const CTYPE* const in_ptr,
    SizesArrayRef in_sizes,
    StridesArrayRef in_strides,
    const CTYPE* const w_ptr,
    SizesArrayRef w_sizes,
    StridesArrayRef w_strides,
    const std::optional<Tensor>& bias,
    const char* const bias_ptr,
    LoadFn load_bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    CTYPE* const out_ptr,
    SizesArrayRef out_sizes,
    StridesArrayRef out_strides,
    const size_t batch,
    const size_t group,
    const size_t out_c,
    bool transposed) {
  if (transposed) {
    transposed_conv3d_impl(
        in_ptr,
        in_sizes,
        in_strides,
        w_ptr,
        w_sizes,
        w_strides,
        bias,
        bias_ptr,
        load_bias,
        stride,
        padding,
        dilation,
        groups,
        out_ptr,
        out_sizes,
        out_strides,
        batch,
        group,
        out_c);
    return;
  }
  size_t in_C = in_sizes[1];

  int64_t out_D = out_sizes[2];
  int64_t in_D = in_sizes[2];
  int64_t w_D = w_sizes[2];

  int64_t out_H = out_sizes[3];
  int64_t in_H = in_sizes[3];
  int64_t w_H = w_sizes[3];

  int64_t out_W = out_sizes[4];
  int64_t in_W = in_sizes[4];
  int64_t w_W = w_sizes[4];

  size_t in_C_per_group = in_C / groups;
  size_t in_c_start = group * in_C_per_group;

  const int64_t stride_z = stride[0];
  const int64_t padding_z = padding[0];
  const int64_t dilation_z = dilation[0];
  const int64_t stride_y = stride[1];
  const int64_t padding_y = padding[1];
  const int64_t dilation_y = dilation[1];
  const int64_t stride_x = stride[2];
  const int64_t padding_x = padding[2];
  const int64_t dilation_x = dilation[2];

  using COMPUTE_T =
      typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
  const COMPUTE_T bias_val = bias_ptr == nullptr
      ? COMPUTE_T{0}
      : load_bias(&bias_ptr[out_c * bias.value().element_size()]);

  for (const auto out_z : c10::irange(out_D)) {
    for (const auto out_y : c10::irange(out_H)) {
      for (const auto out_x : c10::irange(out_W)) {
        COMPUTE_T accum = 0;
        // Depth taps sit outside the channel loop, so lower ranks, whose depth
        // is 1, add no work per channel.
        for (const auto w_z : c10::irange(w_D)) {
          int64_t in_z = stride_z * out_z + dilation_z * w_z - padding_z;
          if (in_z < 0 || in_z >= in_D) {
            continue;
          }
          for (const auto in_c :
               c10::irange(in_c_start, in_c_start + in_C_per_group)) {
            const int64_t in_plane_idx = batch * in_strides[0] +
                in_c * in_strides[1] + in_z * in_strides[2];
            const int64_t w_plane_idx = out_c * w_strides[0] +
                (in_c - in_c_start) * w_strides[1] + w_z * w_strides[2];
            for (const auto w_y : c10::irange(w_H)) {
              int64_t in_y = stride_y * out_y + dilation_y * w_y - padding_y;
              if (in_y < 0 || in_y >= in_H) {
                continue;
              }
              int64_t in_x = stride_x * out_x - padding_x;
              int64_t in_idx =
                  in_plane_idx + in_y * in_strides[3] + in_x * in_strides[4];
              int64_t w_idx = w_plane_idx + w_y * w_strides[3];
              for (int64_t w_x = 0; w_x < w_W; ++w_x) {
                if (in_x >= 0 && in_x < in_W) {
                  const COMPUTE_T in_val = in_ptr[in_idx];
                  const COMPUTE_T w_val = w_ptr[w_idx];
                  accum += in_val * w_val;
                }
                in_x += dilation_x;
                in_idx += dilation_x * in_strides[4];
                w_idx += w_strides[4];
              }
            }
          }
        }
        const int64_t out_idx = batch * out_strides[0] +
            out_c * out_strides[1] + out_z * out_strides[2] +
            out_y * out_strides[3] + out_x * out_strides[4];
        out_ptr[out_idx] = accum + bias_val;
      }
    }
  }
}

// Views t as (N, C, D, H, W) by inserting leading spatial dims of size 1.
void get_3d_view(
    const Tensor& t,
    executorch::aten::SizesType* sizes,
    executorch::aten::StridesType* strides) {
  for (const auto d : c10::irange(5)) {
    sizes[d] = 1;
    strides[d] = 0;
  }
  const ssize_t added_dims = 5 - t.dim();
  for (const auto d : c10::irange(t.dim())) {
    const ssize_t view_d = d < 2 ? d : d + added_dims;
    sizes[view_d] = t.size(d);
    strides[view_d] = t.strides()[d];
  }
}

// Keep dtype-specific loops separate from the scalar-type dispatch.
template <typename CTYPE, typename LoadFn = CTYPE (*)(const void*)>
ET_NOINLINE void convolution_wrapper(
    const Tensor& in,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    LoadFn load_bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    int64_t groups,
    Tensor& out) {
  // A 1-D or 2-D convolution is a 3-D one whose leading spatial dims have size
  // 1, stride 1, padding 0 and dilation 1, so every rank shares the 3-D
  // implementation.
  executorch::aten::SizesType in_sizes[5];
  executorch::aten::StridesType in_strides[5];
  get_3d_view(in, in_sizes, in_strides);
  executorch::aten::SizesType weight_sizes[5];
  executorch::aten::StridesType weight_strides[5];
  get_3d_view(weight, weight_sizes, weight_strides);
  executorch::aten::SizesType out_sizes[5];
  executorch::aten::StridesType out_strides[5];
  get_3d_view(out, out_sizes, out_strides);

  int64_t stride_3d[] = {1, 1, 1};
  int64_t padding_3d[] = {0, 0, 0};
  int64_t dilation_3d[] = {1, 1, 1};
  const ssize_t spatial_dims = in.dim() - 2;
  for (const auto i : c10::irange(spatial_dims)) {
    const ssize_t d = 3 - spatial_dims + i;
    stride_3d[d] = val_at(stride, i);
    padding_3d[d] = val_at(padding, i, /*default_value=*/0);
    dilation_3d[d] = val_at(dilation, i);
  }

  CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();
  const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();
  const CTYPE* const w_ptr = weight.const_data_ptr<CTYPE>();
  const char* const bias_ptr = bias.has_value()
      ? reinterpret_cast<const char*>(bias.value().const_data_ptr())
      : nullptr;

  size_t out_N = out.size(0);
  size_t out_C = out.size(1);
  size_t out_C_per_group = out_C / groups;

  for (const auto batch : c10::irange(out_N)) {
    for (const auto group : c10::irange(groups)) {
      // Align channel offset based on the group
      size_t out_c_start = group * out_C_per_group;
      // Populate all the out channels in the group
      for (const auto out_c :
           c10::irange(out_c_start, out_c_start + out_C_per_group)) {
        conv3d_impl(
            in_ptr,
            in_sizes,
            in_strides,
            w_ptr,
            weight_sizes,
            weight_strides,
            bias,
            bias_ptr,
            load_bias,
            stride_3d,
            padding_3d,
            dilation_3d,
            groups,
            out_ptr,
            out_sizes,
            out_strides,
            batch,
            group,
            out_c,
            transposed);
      }
    }
  }
}

} // namespace

Tensor& convolution_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_convolution_args(
          in,
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_convolution_out_target_size(
      in,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_sizes,
      &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, in.dim() - 2),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.numel() == 0) {
    return out;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "convolution.out");

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    using COMPUTE_T =
        typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
    const auto load_bias = bias.has_value()
        ? utils::internal::get_load_to_compute_fn<COMPUTE_T, name>(
              ctx, bias.value(), utils::SupportedTensorDtypes::REALHBF16)
        : nullptr;
    convolution_wrapper<CTYPE>(
        in,
        weight,
        bias,
        load_bias,
        stride,
        padding,
        dilation,
        transposed,
        groups,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
