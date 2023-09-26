/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Checks sizes passed to `ones_out` (`size_int64_t`) matches those within
 * `out` tensor (`size_int32_t`).
 */
void size_check(
    exec_aten::ArrayRef<int64_t> size_int64_t,
    exec_aten::ArrayRef<int32_t> size_int32_t) {
  ET_CHECK(size_int64_t.size() == size_int32_t.size());
  for (int i = 0; i < size_int64_t.size(); i++) {
    ET_CHECK(((int64_t)size_int32_t[i] == size_int64_t[i]));
  }
};

/**
 * Fills the `out` tensor with value 1.
 */
template <class CTYPE>
void ones_kernel(Tensor& out) {
  // Create pointer over `out` data with `CTYPE`.
  auto data_out = out.mutable_data_ptr<CTYPE>();

  // Set each element of the tensor to the "1" value for the type.
  for (size_t i = 0; i < out.numel(); i++) {
    data_out[i] = static_cast<CTYPE>(1);
  }
};

} // namespace

/**
 * `ones_out` implementation.
 */
Tensor& ones_out(RuntimeContext& ctx, IntArrayRef size, Tensor& out) {
  (void)ctx;
  size_check(size, out.sizes());

#define ONES_OUT(ctype, dtype) \
  case ScalarType::dtype:      \
    ones_kernel<ctype>(out);   \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, ONES_OUT)
    default:
      ET_CHECK_MSG(
          false,
          "out tensor should be a real or bool dtype, but got %" PRId8,
          static_cast<int8_t>(out.scalar_type()));
  }
#undef ONES_OUT

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
