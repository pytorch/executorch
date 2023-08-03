/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>
/**
 * Performs a matrix multiplication of the matrices input and mat2.
 *
 * If input is a (n \times m)(n×m) tensor, mat2 is a (m \times p)(m×p) tensor,
 * out will be a (n \times p)(n×p) tensor.
 *
 * NOTE
 *
 * This function does not broadcast. For broadcasting matrix products, see
 * torch.matmul().
 */

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

/**
 * Asserts that the parameters are valid.
 * self (n x m), mat1 (m x p) and out (n x p)
 * z[i][j] = sum(x[i][k] * y[k][j]), for k in range(m)
 */
void check_mm_out_args(const Tensor& self, const Tensor& mat1, Tensor& out) {
  // Ensure dimensions are the same for all tensors
  ET_CHECK_MSG(
      self.dim() == mat1.dim() && self.dim() == out.dim(),
      "self.dim() %zd and mat1.dim() %zd and out.dim() %zd are not the same",
      self.dim(),
      mat1.dim(),
      out.dim());
  // Ensure dimension is 2 for all tensors
  ET_CHECK_MSG(self.dim() == 2, "self.dim() %zd != 2", self.dim());
  // Ensure 3 tensors are having the same dtype
  ET_CHECK_SAME_DTYPE3(self, mat1, out);
  // Ensure the out size is compatible with input tensors
  ET_CHECK_MSG(
      mat1.size(1) == out.size(1),
      "mat1.size(1) %zd != out.size(1) %zd",
      mat1.size(1),
      out.size(1));
  ET_CHECK_MSG(
      self.size(0) == out.size(0),
      "self.size(0) %zd != out.size(0) %zd",
      self.size(0),
      out.size(0));
}

// for simplicity, assuming all tensors are of the same type. T is the tensor
// dtype.
template <typename T>
Tensor& mm_out_kernel(const Tensor& self, const Tensor& mat1, Tensor& out) {
  const T* self_data = self.const_data_ptr<T>();
  const T* mat1_data = mat1.const_data_ptr<T>();
  T* out_data = out.mutable_data_ptr<T>();

  size_t m = self.size(0);
  size_t n = self.size(1);
  size_t p = out.size(1);

  vec_matmul<T>(out_data, self_data, mat1_data, m, n, p);

  return out;
}

} // namespace

/**
 * mm.out(Tensor self, Tensor mat1, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& mm_out(
    RuntimeContext& context,
    const Tensor& self,
    const Tensor& mat1,
    Tensor& out) {
  (void)context;
  Tensor::SizesType expected_output_size[2];
  expected_output_size[0] = self.size(0);
  expected_output_size[1] = mat1.size(1);
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(out.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  check_mm_out_args(self, mat1, out);
  auto scalar_type = self.scalar_type();
#define MM_TENSOR(ctype, dtype)            \
  case ScalarType::dtype:                  \
    mm_out_kernel<ctype>(self, mat1, out); \
    break;

  switch (scalar_type) {
    ET_FORALL_REAL_TYPES(MM_TENSOR)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", scalar_type);
  }
#undef MM_TENSOR
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
