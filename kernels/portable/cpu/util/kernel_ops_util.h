/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tuple>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

/**
 * Extracts a value at index i from an int array. If the array length is 1, then
 * the first element will be returned regardless of what i is requested to
 * simulate broadcasting.
 */
inline int64_t val_at(IntArrayRef array, size_t i, int64_t default_value = 1) {
  if (array.size() == 1) {
    return array[0];
  } else if (array.size() > 1) {
    return array[i];
  } else {
    return default_value;
  }
}

/**
 * Checks that all elements of an IntArray are greater than or equal to `val`.
 */
bool int_array_all_ge(IntArrayRef array, int64_t val);

bool stride_is_valid(IntArrayRef stride, size_t kernel_ndim, bool allow_empty);

bool padding_is_valid(
    IntArrayRef padding,
    IntArrayRef kernel_size,
    size_t kernel_ndim,
    bool enforce_half_kernel = false);

bool dilation_is_valid(IntArrayRef dilation, size_t kernel_ndim);

bool output_size_is_valid(
    exec_aten::ArrayRef<exec_aten::SizesType> output_size,
    size_t kernel_ndim);

void get_unsqueezed_sizes(
    const Tensor& t,
    int64_t unsqueeze_dim,
    exec_aten::SizesType* sizes_arr,
    size_t& ndim);

void get_unsqueezed_dim_order(
    const Tensor& t,
    exec_aten::DimOrderType unsqueeze_dim,
    exec_aten::DimOrderType* dim_order_arr);

/**
 * Given an input tensor and N-dim kernel parameters, calculates the output size
 * of the N-dim kernel region.
 */
void calculate_kernel_output_sizes(
    const Tensor& in,
    size_t kernel_ndim,
    IntArrayRef kernel_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    bool ceil_mode = false,
    bool transposed = false,
    IntArrayRef output_padding = {});

//
// Utility functions to apply reduction over a N-dimensional kernel window
//

/**
 * Given a 3-D or 4-D tensor, applies a reduction function over a 2D kernel
 * region for at a given batch and channel output index. Note that reduce_fn
 * should return both an accumulator value and an index value; so for example,
 * if reducing using the max() function, this function will track both the
 * maximum value observed as well as the index of the maximum value that was
 * observed. Therefore reduce_fn should follow the signature:
 *
 * ```
 * std::tuple<T, int64_t> reduce_fn(
 *    const T in_val,
 *    const int64_t in_idx,
 *    T accum,
 *    int64_t accum_idx);
 * ```
 *
 * Then, before writing out the accumulator and index values to out_ptr and
 * indices_ptr respectively, accumulator will be mapped with map_fn, which
 * should follow the signature:
 *
 * ```
 * T map_fn(const int64_t count, const T accum);
 * ```
 *
 * The index is a linear index with respect to the 2D plane formed by
 * the height and width axes. So for a tensor of size (N, C, H, W), an element
 * at location (n, c, h, w) will have a index of h * W + w. Although the an
 * index accumulator is tracked, if `indices_ptr` is `nullptr` then it will not
 * be used. Therefore, for reductions that do not care about indices, the index
 * accumulator can be ignored.
 *
 * @param[in] reduce_fn The reduction function used to update accumulator
 * values.
 * @param[in] map_fn The map function used to post-process the accumulator
 * value before writing to out_ptr.
 * @param[in] include_pad Indicates if the reduction function should be applied
 * over the padded regions implicitly added by the `padding` argument.
 * @param[in] in_ptr The pointer to the input tensor data.
 * @param[in] in_sizes Sizes array describing the size of the input tensor.
 * @param[in] in_strides Strides array describing the strides of the input
 * tensor.
 * @param[in] kernel_size 2D array describing the height and width of the kernel
 * region.
 * @param[in] stride 2D array describing how the kernel region "traverses" over
 * the input tensor.
 * @param[in] padding 2D array describing padding to apply to the input tensor.
 * @param[in] dilation 2D array describing the dilation to apply to the kernel
 * region.
 * @param[in] out_ptr The pointer to the output tensor data.
 * @param[in] out_sizes Sizes array describing the size of the output tensor.
 * @param[in] out_strides Strides array describing the strides of the output
 * @param[in] indices_ptr The pointer to the indices tensor data. Can be
 * `nullptr`.
 * @param[in] batch The batch index of the output locations being computed.
 * @param[in] out_c The channels index of the output locations being computed.
 */
template <typename CTYPE, typename ReduceOp, typename MapOp>
void kernel_reduction_then_map_2d(
    const ReduceOp& reduce_fn,
    const MapOp& map_fn,
    const bool include_pad,
    const CTYPE* const in_ptr,
    const exec_aten::ArrayRef<exec_aten::SizesType> in_sizes,
    const exec_aten::ArrayRef<exec_aten::StridesType> in_strides,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    CTYPE* const out_ptr,
    const exec_aten::ArrayRef<exec_aten::SizesType> out_sizes,
    const exec_aten::ArrayRef<exec_aten::StridesType> out_strides,
    int64_t* const indices_ptr,
    const size_t batch,
    const size_t out_c) {
  size_t in_dim = in_sizes.size();
  size_t out_dim = out_sizes.size();

  size_t out_H = out_sizes[in_dim - 2];
  size_t in_H = in_sizes[in_dim - 2];

  size_t out_W = out_sizes[in_dim - 1];
  size_t in_W = in_sizes[in_dim - 1];

  exec_aten::SizesType in_coord[kTensorDimensionLimit];
  exec_aten::SizesType out_coord[kTensorDimensionLimit];
  if (in_dim == 4) {
    in_coord[0] = batch;
    out_coord[0] = batch;
  }
  in_coord[in_dim - 3] = out_c;
  out_coord[in_dim - 3] = out_c;

  int64_t k_H = val_at(kernel_size, 0);
  int64_t k_W = val_at(kernel_size, 1);
  int64_t s_H = val_at(stride, 0, /*default_value=*/k_H);
  int64_t s_W = val_at(stride, 1, /*default_value=*/k_W);
  int64_t p_H = val_at(padding, 0, /*default_value=*/0);
  int64_t p_W = val_at(padding, 1, /*default_value=*/0);
  int64_t d_H = val_at(dilation, 0, /*default_value=*/1);
  int64_t d_W = val_at(dilation, 1, /*default_value=*/1);

  // Compute 2D output region
  for (size_t out_y = 0; out_y < out_H; ++out_y) {
    out_coord[in_dim - 2] = out_y;
    for (size_t out_x = 0; out_x < out_W; ++out_x) {
      out_coord[in_dim - 1] = out_x;

      bool accum_initialized = false;
      CTYPE accum = 0;
      int64_t accum_idx = 0;
      int64_t count = 0;

      int64_t ih0 = out_y * s_H - p_H;
      int64_t iw0 = out_x * s_W - p_W;
      int64_t ih1 = std::min(ih0 + k_H, static_cast<int64_t>(in_H) + p_H);
      int64_t iw1 = std::min(iw0 + k_W, static_cast<int64_t>(in_W) + p_W);
      int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
      ih0 = std::max(ih0, (int64_t)0);
      iw0 = std::max(iw0, (int64_t)0);
      ih1 = std::min(ih1, static_cast<int64_t>(in_H));
      iw1 = std::min(iw1, static_cast<int64_t>(in_W));

      if (ih0 >= ih1 || iw0 >= iw1) {
        continue;
      }

      if (include_pad) {
        count = pool_size;
      } else {
        count = (ih1 - ih0) * (iw1 - iw0);
      }

      for (size_t w_y = 0; w_y < k_H; ++w_y) {
        int64_t stride_y = s_H;
        int64_t padding_y = p_H;
        int64_t dilation_y = d_H;

        size_t in_y = stride_y * out_y + dilation_y * w_y - padding_y;
        in_coord[in_dim - 2] = in_y;

        for (size_t w_x = 0; w_x < k_W; ++w_x) {
          int64_t stride_x = s_W;
          int64_t padding_x = p_W;
          int64_t dilation_x = d_W;

          size_t in_x = stride_x * out_x + dilation_x * w_x - padding_x;
          in_coord[in_dim - 1] = in_x;

          const bool x_in_bound = (in_x >= 0 && in_x < in_W);
          const bool y_in_bound = (in_y >= 0 && in_y < in_H);
          const bool xy_in_bound = (x_in_bound && y_in_bound);

          CTYPE in_val = 0;
          if (xy_in_bound) {
            size_t in_idx =
                calculate_linear_index(in_coord, in_strides.data(), in_dim);
            in_val = in_ptr[in_idx];
          }

          int64_t idx = in_y * in_W + in_x;
          if (include_pad) {
            idx =
                in_y + padding_y * (in_W + 2 * padding_x) + (in_x + padding_x);
          }

          if (xy_in_bound) {
            if (!accum_initialized) {
              accum = in_val;
              accum_idx = idx;
              accum_initialized = true;
            } else {
              std::tuple<CTYPE, int64_t> ret =
                  reduce_fn(in_val, idx, accum, accum_idx);
              accum = std::get<0>(ret);
              accum_idx = std::get<1>(ret);
            }
          }
        }
      }

      size_t out_idx =
          calculate_linear_index(out_coord, out_strides.data(), out_dim);
      out_ptr[out_idx] = map_fn(count, accum);
      if (indices_ptr) {
        indices_ptr[out_idx] = accum_idx;
      }
    }
  }
}

/**
 * Given a 3-D {C, H, W} or 4-D {N, C, H, W} tensor, applies a reduction
 * function over a 2D kernel region, which will return two accumulator values
 * (the first associated with the reduced value and the second associated with
 * an input index). Then apply a map function to the first accumulator value
 * before writing to out. Optionally, the second accumulator value will be
 * written to indices, if provided.
 *
 * reduce_fn should have the following
 * signature:
 *
 * ```
 * std::tuple<T, int64_t> reduce_fn(
 *    const T in_val,
 *    const int64_t in_idx,
 *    const T accum,
 *    const int64_t accum_idx);
 * ```
 *
 * map_fn should have the following signature:
 *
 * ```
 * T map_fn(const int64_t count, const T accum);
 * ```
 *
 * TODO(ssjia) Allow this to handle 1-D kernels as well by unsqueezing
 * appropriately.
 *
 * @param[in] reduce_fn The reduction function used to update accumulator
 * values.
 * @param[in] map_fn The map function used to post-process the accumulated
 * value before writing to out.
 * @param[in] include_pad Indicates if the reduction function should be applied
 * over the padded regions implicitly added by the `padding` argument.
 * @param[in] in The input tensor.
 * @param[in] kernel_size 2D array describing the height and width of the kernel
 * region.
 * @param[in] stride 2D array describing how the kernel region "traverses" over
 * the input tensor.
 * @param[in] padding 2D array describing padding to apply to the input tensor.
 * @param[in] dilation 2D array describing the dilation to apply to the kernel
 * region.
 * @param[in] out The output tensor.
 * @param[in] indices An optional indices output tensor to write out to.
 */
template <typename CTYPE, typename ReduceOp, typename MapOp>
void apply_kernel_2d_reduce_then_map_fn(
    const ReduceOp& reduce_fn,
    const MapOp& map_fn,
    const bool include_pad,
    const Tensor& in,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    Tensor& out,
    exec_aten::optional<Tensor> indices = {}) {
  exec_aten::ArrayRef<exec_aten::SizesType> in_sizes = in.sizes();
  exec_aten::ArrayRef<exec_aten::SizesType> out_sizes = out.sizes();

  exec_aten::ArrayRef<exec_aten::DimOrderType> in_dim_order = in.dim_order();
  exec_aten::ArrayRef<exec_aten::DimOrderType> out_dim_order = out.dim_order();

  exec_aten::StridesType in_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      in_sizes.data(), in_dim_order.data(), in_sizes.size(), in_strides);

  exec_aten::StridesType out_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      out_sizes.data(), out_dim_order.data(), out_sizes.size(), out_strides);

  CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();
  const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();

  int64_t* indices_ptr = nullptr;
  if (indices.has_value()) {
    indices_ptr = indices.value().mutable_data_ptr<int64_t>();
  }

  size_t batch_size = 1;
  if (in.dim() == 4) {
    batch_size = in_sizes[0];
  }
  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t channel = 0; channel < in_sizes[in.dim() - 3]; ++channel) {
      kernel_reduction_then_map_2d(
          reduce_fn,
          map_fn,
          include_pad,
          in_ptr,
          in_sizes,
          {in_strides, 4},
          kernel_size,
          stride,
          padding,
          dilation,
          out_ptr,
          out_sizes,
          {out_strides, 4},
          indices_ptr,
          batch,
          channel);
    }
  }
}

//
// Operator specific utility functions
//

bool check_arange_args(double start, double end, double step, Tensor& out);

bool check_avg_pool2d_args(
    const Tensor& in,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const bool ceil_mode,
    const bool count_include_pad,
    const exec_aten::optional<int64_t>& divisor_override,
    const Tensor& out);

void get_avg_pool2d_out_target_size(
    const Tensor& in,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const bool ceil_mode,
    exec_aten::SizesType* const out_sizes,
    size_t* const out_ndim);

bool check_convolution_args(
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    const Tensor& out);

void get_convolution_out_target_size(
    const Tensor& in,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_cumsum_args(
    const Tensor& self,
    int64_t dim,
    optional<ScalarType> enforced_dtype,
    Tensor& out);

bool check_max_pool2d_with_indices_args(
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices);

void get_max_pool2d_with_indices_out_target_size(
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_masked_fill_args(
    const Tensor& in,
    const Tensor& mask,
    const Scalar& value,
    Tensor& out);

bool check_constant_pad_args(
    const Tensor& in,
    IntArrayRef pad,
    const Scalar& value,
    Tensor& out);

Error resize_constant_pad_output(
    const Tensor& in,
    IntArrayRef pad,
    Tensor& out);

bool check_embedding_args(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& out);

Error resize_embedding_output(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& out);

bool check_alpha_type(
    const ScalarType alpha_type,
    const ScalarType common_type);

} // namespace executor
} // namespace torch
