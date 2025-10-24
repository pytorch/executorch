/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>
#include <executorch/backends/cadence/fusion_g3/operators/xt_utils.h>

#include <cstring>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ArrayRef;
using ::executorch::aten::ScalarType;
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

Tensor& cat_out(
    KernelRuntimeContext& ctx,
    ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  if (dim < 0) {
    dim += out.dim();
  }

  int kTensorDimensionLimit = executorch::runtime::kTensorDimensionLimit;

#ifdef OP_ARG_CHECK

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  torch::executor::get_cat_out_target_size(
      tensors, dim, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(
          out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);
#endif
  // Special handling when all inputs are 1D-empty tensors for aten
  // consistency In that case, just return an 1D-empty tensor without checking
  // dim
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

  const signed char* inp_tensors[tensors.size()];
  const int* inp_tensors_shapes[tensors.size()];

  int inp_shapes_size[tensors.size()];

  int temp_sizes[tensors.size()][kTensorDimensionLimit];
  ArrayRef<Tensor::SizesType> temp_size;

  for (int i = 0; i < tensors.size(); i++) {
    inp_tensors[i] = tensors[i].const_data_ptr<signed char>();
    temp_size = tensors[i].sizes();

    for (int j = 0; j < temp_size.size(); j++) {
      temp_sizes[i][j] = temp_size[j];
    }
    inp_tensors_shapes[i] = temp_sizes[i]; // input shapes
    inp_shapes_size[i] = temp_size.size(); // number of input dimensions
  }

  signed char* out_data = out.mutable_data_ptr<signed char>();

  const ArrayRef<Tensor::SizesType> out_size = out.sizes();
  int out_shapes[kTensorDimensionLimit];
  for (int i = 0; i < out_size.size(); i++) // output shapes
  {
    out_shapes[i] = out_size[i];
  }

  bool optimized = true;

  for (int i = 0; i < tensors.size(); i++) {
    if (out.scalar_type() != tensors[i].scalar_type()) {
      optimized = false;
      break;
    }
  }

  if ((optimized) && (out.scalar_type() == ScalarType::Int) ||
      (out.scalar_type() == ScalarType::Short) ||
      (out.scalar_type() == ScalarType::Char) ||
      (out.scalar_type() == ScalarType::UInt32) ||
      (out.scalar_type() == ScalarType::UInt16) ||
      (out.scalar_type() == ScalarType::Byte) ||
      (out.scalar_type() == ScalarType::Float)) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_cat,
        out_data,
        out_shapes,
        inp_tensors,
        inp_tensors_shapes,
        inp_shapes_size[0],
        tensors.size(),
        (int)dim,
        get_element_size(out.scalar_type()));
  } else {
    ET_KERNEL_CHECK(
        ctx,
        torch::executor::check_cat_args(tensors, dim, out),
        InvalidArgument,
        out);

    const size_t outer = executorch::runtime::getLeadingDims(out, dim);
    const size_t dim_stride = executorch::runtime::getTrailingDims(out, dim);
    const size_t ninputs = tensors.size();

    const auto out_type = out.scalar_type();
    ET_SWITCH_REALHB_TYPES(out_type, ctx, "cat.out", CTYPE_OUT, [&] {
      CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t i = 0; i < outer; ++i) {
        for (size_t j = 0; j < ninputs; ++j) {
          const auto in_type = tensors[j].scalar_type();
          ET_SWITCH_REALHB_TYPES(in_type, ctx, "cat.out", CTYPE_IN, [&] {
            if (tensors[j].numel() == 0) {
              return;
            }
            size_t inner = tensors[j].size(dim) * dim_stride;
            const CTYPE_IN* const in_ptr =
                tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;

            for (size_t k = 0; k < inner; ++k) {
              out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
            }
            out_ptr += inner;
          });
        }
      }
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
