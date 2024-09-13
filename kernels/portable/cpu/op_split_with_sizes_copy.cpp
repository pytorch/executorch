/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

void split_with_sizes_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing. Note that this op does not accept 0
  // dimensional input tensors.
  if (dim < 0) {
    dim += in.dim();
  }

  ET_KERNEL_CHECK(
      ctx,
      check_split_with_sizes_copy_args(in, split_sizes, dim, out),
      InvalidArgument, );

  for (size_t i = 0; i < out.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx, tensors_have_same_dim_order(in, out[i]), InvalidArgument, );
  }

  // If out is empty, then nothing needs to be done after checking the args.
  // Valid args implies that in.size(dim) == 0 and split_sizes is also empty.
  if (out.size() == 0) {
    return;
  }

  // Check that all chunks broadcast to their respective out tensor
  Tensor::SizesType target_out_sizes[kTensorDimensionLimit];
  size_t target_out_ndim = in.dim();
  for (size_t d = 0; d < in.dim(); ++d) {
    target_out_sizes[d] = static_cast<Tensor::SizesType>(in.size(d));
  }

  for (size_t i = 0; i < split_sizes.size(); i++) {
    target_out_sizes[dim] = static_cast<Tensor::SizesType>(split_sizes[i]);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(out[i], {target_out_sizes, target_out_ndim}) == Error::Ok,
        InvalidArgument, );
  }

  const size_t leading_dims = getLeadingDims(in, dim);
  const size_t trailing_dims = getTrailingDims(in, dim);
  const size_t step = in.size(dim) * trailing_dims;

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE_IN, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, __func__, CTYPE_OUT, [&]() {
      const CTYPE_IN* in_data = in.const_data_ptr<CTYPE_IN>();

      // Iterate through list of out tensors
      for (size_t i = 0; i < out.size(); ++i) {
        const Tensor& out_tensor = out[i];

        // If out tensor is empty, no action is required
        if (out_tensor.numel() == 0) {
          continue;
        }

        size_t chunk_step = split_sizes[i] * trailing_dims;

        // Update target out shape
        target_out_sizes[dim] = static_cast<Tensor::SizesType>(split_sizes[i]);
        ArrayRef<Tensor::SizesType> target_shape(
            {target_out_sizes, target_out_ndim});

        // Check if output involves broadcasting
        const bool is_broadcasted = !out_tensor.sizes().equals(target_shape);

        CTYPE_OUT* out_data = out_tensor.mutable_data_ptr<CTYPE_OUT>();

        // Simpler logic if there's no broadcasting
        if (!is_broadcasted) {
          const CTYPE_IN* src = in_data;
          for (size_t j = 0; j < leading_dims; ++j) {
            for (size_t k = 0; k < chunk_step; ++k) {
              out_data[k] = convert<CTYPE_OUT, CTYPE_IN>(src[k]);
            }
            src += step;
            out_data += chunk_step;
          }
        } else { // Otherwise, we need to do a copy with broadcasting
          // Compute target strides
          Tensor::StridesType target_out_strides[kTensorDimensionLimit];
          target_out_strides[in.dim() - 1] = 1;
          for (int d = in.dim() - 2; d >= 0; --d) {
            target_out_strides[d] = target_out_strides[d + 1] *
                static_cast<Tensor::StridesType>(target_out_sizes[d + 1]);
          }
          ArrayRef<Tensor::StridesType> target_strides(
              {target_out_strides, target_out_ndim});

          // For each element in the out tensor, find its corresponding index
          // in the input tensor and copy it over
          for (size_t ix = 0; ix < out_tensor.numel(); ++ix) {
            size_t out_coord[kTensorDimensionLimit];
            delinearize_index(ix, out_tensor, out_coord, kTensorDimensionLimit);

            size_t in_linear_index = linearize_access_indexes(
                out_coord, out_tensor.dim(), target_shape, target_strides);

            out_data[ix] =
                convert<CTYPE_OUT, CTYPE_IN>(in_data[in_linear_index]);
          }
        }

        // Move input data pointer
        in_data += chunk_step;
      }
    });
  });
}

} // namespace native
} // namespace executor
} // namespace torch
