/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>
#include <executorch/backends/cadence/fusion_g3/operators/xt_utils.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ArrayRef;
using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

/* ScalarType in Executorch do not have support for below data types.
 * So, creating a placeholder for these data types. Once, ScalarTypes is
 * updated to have support for below data types, these can be removed and
 * operator need to be updated accordingly
 */

namespace impl {
namespace G3 {
namespace native {

namespace {

void increment_coordinate_permuted(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = dims.size() - 1; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (coordinate[d] == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}

} // namespace

Tensor& permute_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;
  int kTensorDimensionLimit = 5;
  /* if the arguments are passed properly to the operator disable the Macro -
   * "OP_ARG_CHECK" if not the case, enable the Macro - "OP_ARG_CHECK", to have
   * the checks only in operator level(As there are no checks in kernel).
   */
#ifdef OP_ARG_CHECK
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  torch::executor::get_permute_copy_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(
          out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);
#endif

  const ArrayRef<Tensor::SizesType> in_size = in.sizes();
  const ArrayRef<Tensor::SizesType> out_size = out.sizes();

  int inp_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];
  int permute_vec[kTensorDimensionLimit];

  /* input shapes and output shapes */
  for (auto i = 0; i < in_size.size(); i++) {
    inp_shape[i] = in_size[i];
  }

  for (auto i = 0; i < out_size.size(); i++) {
    out_shape[i] = out_size[i];
  }

  for (int i = 0; i < in.dim(); i++) {
    permute_vec[i] = (int)dims[i];
  }
  signed char* out_data = out.mutable_data_ptr<signed char>();
  const signed char* const inp_data = in.const_data_ptr<signed char>();

  if (((out.scalar_type() == in.scalar_type()) &&
           (out.scalar_type() == ScalarType::Int) ||
       (out.scalar_type() == ScalarType::Short) ||
       (out.scalar_type() == ScalarType::Char) ||
       (out.scalar_type() == ScalarType::UInt32) ||
       (out.scalar_type() == ScalarType::UInt16) ||
       (out.scalar_type() == ScalarType::Byte) ||
       (out.scalar_type() == ScalarType::Float)) &&
      (in.dim() <= 5)) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_permute,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        permute_vec,
        in.dim(),
        get_element_size(out.scalar_type()));
  } else {
    ET_KERNEL_CHECK(
        ctx,
        torch::executor::check_permute_copy_args(in, dims, out),
        InvalidArgument,
        out);

    const auto in_type = out.scalar_type();
    size_t in_coord[executorch::runtime::kTensorDimensionLimit] = {0};
    size_t trailing_dims_memo[executorch::runtime::kTensorDimensionLimit];
    executorch::runtime::memoizeTrailingDims(in, trailing_dims_memo);
    // in and out must be the same dtype
    ET_SWITCH_ALL_TYPES(in_type, ctx, "permute_copy.out", CTYPE, [&] {
      const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
      CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

      for (size_t i = 0; i < out.numel(); ++i) {
        out_data[i] =
            in_data[executorch::runtime::coordinateToIndexWithTrailingDimsMemo(
                in, in_coord, trailing_dims_memo)];
        increment_coordinate_permuted(in, in_coord, dims);
      }
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
