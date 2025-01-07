/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/fusion_g3/operators/xt_macros.h>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::runtime::KernelRuntimeContext;
using SizesType = ::executorch::aten::SizesType;
using Tensor = ::executorch::aten::Tensor;
using IntArrayRef = ::executorch::aten::ArrayRef<int64_t>;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::runtime::Error;

/* ScalarType in Executorch do not have support for below data types.
 * So, creating a placeholder for these data types. Once, ScalarTypes is
 * updated to have support for below data types, these can be removed and
 * operator need to be updated accordingly
 */
enum datatype {
  Ushort = 20,
  Uint = 23,
};

namespace cadence {
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
      torch::executor::check_permute_copy_args(in, dims, out),
      InvalidArgument,
      out);

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

  const ::executorch::aten::ArrayRef<Tensor::SizesType> in_size = in.sizes();
  const ::executorch::aten::ArrayRef<Tensor::SizesType> out_size = out.sizes();

  int inp_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  /* input shapes and output shapes */
  for (auto i = 0; i < in_size.size(); i++) {
    inp_shape[i] = in_size[i];
  }

  for (auto i = 0; i < out_size.size(); i++) {
    out_shape[i] = out_size[i];
  }

  int permute_vec[in.dim()];
  for (int i = 0; i < in.dim(); i++) {
    permute_vec[i] = (int)dims[i];
  }
  signed char* out_data = out.mutable_data_ptr<signed char>();
  const signed char* const inp_data = in.const_data_ptr<signed char>();

  if ((out.scalar_type() == ScalarType::Int) && (in.dim() <= 5)) {
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
        sizeof(int));
  } else if ((out.scalar_type() == ScalarType::Short) && (in.dim() <= 5)) {
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
        sizeof(short));
  } else if ((out.scalar_type() == ScalarType::Char) && (in.dim() <= 5)) {
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
        sizeof(char));

  } else if ((out.scalar_type() == (ScalarType)Uint) && (in.dim() <= 5)) {
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
        sizeof(int));
  } else if ((out.scalar_type() == (ScalarType)Ushort) && (in.dim() <= 5)) {
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
        sizeof(short));
  } else if ((out.scalar_type() == ScalarType::Byte) && (in.dim() <= 5)) {
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
        sizeof(char));
  } else {
    const auto in_type = out.scalar_type();
    size_t in_coord[5] = {0};
    size_t trailing_dims_memo[kTensorDimensionLimit];
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
} // namespace cadence