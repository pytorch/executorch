// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

void check_stack_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_CHECK(tensors.size() > 0);

  // All input tensors need to be of the same size
  // https://pytorch.org/docs/stable/generated/torch.stack.html
  for (size_t i = 0; i < tensors.size(); i++) {
    // All input dtypes must be castable to the output dtype.
    ET_CHECK(canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_CHECK(tensors[i].dim() == tensors[0].dim());
    for (size_t d = 0; d < tensors[i].dim(); d++) {
      ET_CHECK(tensors[i].size(d) == tensors[0].size(d));
    }
  }

  // The output tensor will have a dimension inserted, so dim should be between
  // 0 and ndim_of_inputs + 1
  ET_CHECK(dim >= 0 && dim < tensors[0].dim() + 1);
}

void get_stack_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = tensors[0].dim() + 1;

  for (size_t d = 0; d < *out_ndim; ++d) {
    if (d < dim) {
      out_sizes[d] = tensors[0].size(d);
    } else if (d == dim) {
      out_sizes[d] = tensors.size();
    } else {
      out_sizes[d] = tensors[0].size(d - 1);
    }
  }
}

} // namespace executor
} // namespace torch
