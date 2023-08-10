/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

// A helper function for comparsion to make sure the info the error msg shows in
// check function is in line with what user enters to `unsqueeze_copy_out`
int64_t compare_with_dim(int64_t d, int64_t dim, int64_t out_dim) {
  if (dim < 0) {
    dim += out_dim;
  }
  return d - dim;
}

void check_and_update_unsqueeze_copy_out_args(
    const Tensor input,
    int64_t dim,
    const Tensor out) {
  // The input and out shall share same dtype
  ET_CHECK_SAME_DTYPE2(input, out);

  ET_CHECK_MSG(
      dim >= -out.dim() && dim < out.dim(),
      "dim %" PRId64 " out of range [-%zd,%zd)",
      dim,
      out.dim(),
      out.dim());

  // The shape of input and out shall obey the relationship:
  // 1. input.dim() == out.dim()-1
  // 2. input.size(i) == out.size(i) for all i < dim
  // 3. input.size(i-1) == out.size(i) for all i >= dim
  // 4. out.size(dim) == 1
  ET_CHECK(input.dim() == out.dim() - 1);

  for (size_t d = 0; d < out.dim(); d++) {
    int64_t compared = compare_with_dim(d, dim, out.dim());
    if (compared < 0) {
      // d < dim
      ET_CHECK_MSG(
          input.size(d) == out.size(d),
          "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
          d,
          input.size(d),
          d,
          out.size(d),
          dim);
    } else if (compared > 0) {
      // d > dim
      ET_CHECK_MSG(
          input.size(d - 1) == out.size(d),
          "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
          d - 1,
          input.size(d),
          d,
          out.size(d),
          dim);
    } else { // d == dim
      ET_CHECK_MSG(
          out.size(d) == 1,
          "out.size(%zu) %zd shall equal 1 | dim = %" PRId64,
          d,
          out.size(d),
          dim);
    }
  }
}
} // namespace

// unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
// -> Tensor(a!)
Tensor& unsqueeze_copy_out(
    RuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  (void)ctx;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  // I think this is safe to do but need to confirm.
  // If we can do this then subsequent checks that specialize on dim < 0
  // are not needed
  if (dim < 0) {
    dim += out.dim();
  }
  for (size_t i = 0; i < out.dim(); ++i) {
    if (i < dim) {
      expected_output_size[i] = self.size(i);
    } else if (i > dim) {
      expected_output_size[i] = self.size(i - 1);
    } else {
      expected_output_size[i] = 1;
    }
  }
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(out.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  check_and_update_unsqueeze_copy_out_args(/*input=*/self, dim, out);

  if (self.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
