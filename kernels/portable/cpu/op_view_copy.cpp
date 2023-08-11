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
void size_compare(
    exec_aten::ArrayRef<int64_t> size_int64_t,
    exec_aten::ArrayRef<int32_t> size_int32_t) {
  bool size_inferred = false;
  for (int i = 0; i < size_int64_t.size(); i++) {
    // If this value is -1 it implies that this dimension is inferred.
    if (size_int64_t[i] == -1) {
      ET_CHECK_MSG(!size_inferred, "Multiple dimensions cannot be inferred.");
      size_inferred = true;
    }
    ET_CHECK(
        ((int64_t)size_int32_t[i] == size_int64_t[i]) ||
        (size_int64_t[i] == -1));
  }
}
} // namespace

// view_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor& view_copy_out(
    RuntimeContext& ctx,
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out) {
  (void)ctx;
  ET_CHECK(size_int64_t.size() == out.sizes().size());
  Tensor::SizesType expected_output_size[16];
  size_t out_numels_without_minus_1 = 1;
  int32_t minus_1_dim = -1;
  for (size_t i = 0; i < out.dim(); ++i) {
    if (size_int64_t[i] != -1) {
      expected_output_size[i] = static_cast<Tensor::SizesType>(size_int64_t[i]);
      out_numels_without_minus_1 = out_numels_without_minus_1 * size_int64_t[i];
    } else {
      // TODO(kimishpatel): Add test to hit this line
      ET_CHECK_MSG(minus_1_dim == -1, "At most one view copy dim can be -1.");
      minus_1_dim = i;
    }
  }
  if (minus_1_dim >= 0) {
    expected_output_size[minus_1_dim] =
        self.numel() / out_numels_without_minus_1;
  }
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(out.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  // The input and out shall share same dtype and numel
  ET_CHECK(self.numel() == out.numel());
  ET_CHECK_SAME_DTYPE2(self, out);

  // The size of out should equal target size.
  size_compare(size_int64_t, out.sizes());

  if (self.nbytes() > 0) {
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
