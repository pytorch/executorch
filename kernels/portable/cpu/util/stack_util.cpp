/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/stack_util.h>

namespace torch::executor::native {

bool check_stack_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_LOG_AND_RETURN_IF_FALSE(tensors.size() > 0);

  // All input tensors need to be of the same size
  // https://pytorch.org/docs/stable/generated/torch.stack.html
  for (const auto i : c10::irange(tensors.size())) {
    // All input dtypes must be castable to the output dtype.
    ET_LOG_AND_RETURN_IF_FALSE(
        canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(tensors[i], tensors[0].dim()));
    for (const auto d : c10::irange(tensors[i].dim())) {
      ET_LOG_AND_RETURN_IF_FALSE(
          tensors_have_same_size_at_dims(tensors[i], d, tensors[0], d));
    }
  }

  // The output tensor will have a dimension inserted, so dim should be between
  // 0 and ndim_of_inputs + 1
  ET_LOG_AND_RETURN_IF_FALSE(dim >= 0 && dim < tensors[0].dim() + 1);

  return true;
}

void get_stack_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = tensors[0].dim() + 1;

  for (const auto d : c10::irange(*out_ndim)) {
    int64_t d_ = static_cast<int64_t>(d);
    if (d_ < dim) {
      out_sizes[d_] = tensors[0].size(d_);
    } else if (d_ == dim) {
      out_sizes[d_] = tensors.size();
    } else {
      out_sizes[d_] = tensors[0].size(d_ - 1);
    }
  }
}

Tensor& stack_out_impl(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)ctx;
  const size_t outer = getLeadingDims(out, dim);
  const size_t inner = getTrailingDims(out, dim);
  const size_t ninputs = tensors.size();

  const auto out_type = out.scalar_type();
  ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, "stack.out", CTYPE_OUT, [&] {
    CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
    for (size_t i = 0; i < outer; ++i) {
      for (size_t j = 0; j < ninputs; ++j) {
        const auto in_type = tensors[j].scalar_type();
        ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "stack.out", CTYPE_IN, [&] {
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

  return out;
}

} // namespace torch::executor::native
