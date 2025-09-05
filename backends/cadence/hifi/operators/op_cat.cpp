/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::getTrailingDims;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::check_cat_args;
using torch::executor::Error;
using torch::executor::get_cat_out_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& cat_out(
    RuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  if (dim < 0) {
    dim += out.dim();
  }

  ET_KERNEL_CHECK(ctx, check_cat_args(tensors, dim, out), Internal, out);

  Tensor::SizesType
      expected_out_size[executorch::runtime::kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_cat_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  // Special handling when all inputs are 1D-empty tensors for aten consistency
  // In that case, just return an 1D-empty tensor without checking dim
  bool all_1d_empty = true;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() != 0 || tensors[i].dim() != 1) {
      all_1d_empty = false;
      break;
    }
  }
  if (all_1d_empty) {
    return out;
  }

  constexpr auto name = "cat.out";
  constexpr int kNnlibMaxDim = 16;

  bool optimized = true;

  if ((out.scalar_type() != ScalarType::Float) &&
      (out.scalar_type() != ScalarType::Int))
    optimized = false;

  if (optimized) {
    WORD32 num_inp = tensors.size();
    WORD32 num_inp_dims = out.dim();
    WORD32 num_out_dims = num_inp_dims;
    WORD32 axis = dim;

    WORD32 inp_shape[kNnlibMaxDim][kNnlibMaxDim];
    WORD32 p_out_shape[kNnlibMaxDim];

    WORD32* ptr_shape[kNnlibMaxDim];
    const WORD32* ptr[kNnlibMaxDim];

    int k = 0;
    for (int i = 0; i < num_inp; i++) {
      if (tensors[i].numel() == 0)
        continue;
      ptr[k] = (const WORD32*)tensors[i].const_data_ptr<float>();
      for (int j = 0; j < num_inp_dims; j++) {
        inp_shape[k][j] = tensors[i].size(j);
      }
      ptr_shape[k] = inp_shape[k];
      k++;
    }

    num_inp = k;

    for (int i = 0; i < num_out_dims; i++) {
      p_out_shape[i] = out.size(i);
    }

    const WORD32** pp_inps = &ptr[0];

    WORD32* p_out = (WORD32*)out.mutable_data_ptr<float>();

    const WORD32* const* pp_inps_shape = (const WORD32* const*)&ptr_shape[0];

    WORD32 ret_val = xa_nn_concat_32_32(
        p_out,
        p_out_shape,
        pp_inps,
        pp_inps_shape,
        num_out_dims,
        num_inp,
        num_inp_dims,
        axis);

    ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

    return out;
  }

  const size_t outer = getLeadingDims(out, dim);
  const size_t dim_stride = getTrailingDims(out, dim);
  const size_t ninputs = tensors.size();
  const size_t element_size = out.element_size();
  char* out_ptr = static_cast<char*>(out.mutable_data_ptr());

  for (size_t i = 0; i < outer; ++i) {
    for (size_t j = 0; j < ninputs; ++j) {
      if (tensors[j].numel() == 0) {
        continue;
      }
      size_t inner_elements = tensors[j].size(dim) * dim_stride;
      size_t contiguous_bytes = inner_elements * element_size;

      const char* const in_ptr =
          static_cast<const char*>(tensors[j].const_data_ptr()) +
          i * contiguous_bytes;

      std::memcpy(out_ptr, in_ptr, contiguous_bytes);
      out_ptr += contiguous_bytes;
    }
  }

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
