/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

// Performs a batch matrix-matrix product of matrices stored in input and mat2.

// input and mat2 must be 3-D tensors each containing the same number of
// matrices.

// If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m
// \times p)(b×m×p) tensor, out will be a (b \times n \times p)(b×n×p) tensor.

// Note: This function does not broadcast. For broadcasting matrix products, see
// matmul().
namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

namespace {

// Verifies that the parameters are valid.
bool check_bmm_out_args(const Tensor& self, const Tensor& mat2, Tensor& out) {
  // Ensure dimensions is 3 for all input and out
  ET_CHECK_OR_RETURN_FALSE(
      self.dim() == mat2.dim(),
      "self.dim() %zd != mat2.dim() %zd",
      self.dim(),
      mat2.dim());
  ET_CHECK_OR_RETURN_FALSE(
      self.dim() == out.dim(),
      "self.dim() %zd != out.dim() %zd",
      self.dim(),
      out.dim());
  ET_CHECK_OR_RETURN_FALSE(self.dim() == 3, "self.dim() %zd != 3", self.dim());
  // Ensure batch larger than or equals to 0
  ET_CHECK_OR_RETURN_FALSE(
      self.size(0) >= 0, "self.size(0) %zd < 0", self.size(0));
  // Ensure batches are the same
  ET_CHECK_OR_RETURN_FALSE(
      self.size(0) == mat2.size(0),
      "self.size(0) %zd != mat2.size(0) %zd",
      self.size(0),
      mat2.size(0));
  ET_CHECK_OR_RETURN_FALSE(
      self.size(0) == out.size(0),
      "self.size(0) %zd != out.size(0) %zd",
      self.size(0),
      out.size(0));
  // Ensure the out size is compatible with input tensors
  ET_CHECK_OR_RETURN_FALSE(
      mat2.size(2) == out.size(2),
      "mat2.size(2) %zd != out.size(2) %zd",
      mat2.size(2),
      out.size(2));
  ET_CHECK_OR_RETURN_FALSE(
      self.size(1) == out.size(1),
      "self.size(1) %zd != out.size(1) %zd",
      self.size(1),
      out.size(1));

  // Ensure that all tensors share a dtype
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(self, mat2, out));

  return true;
}

template <typename CTYPE>
void bmm_kernel(const Tensor& self, const Tensor& mat2, Tensor& out) {
  using executorch::cpublas::TransposeType;

  if (self.numel() == 0 || mat2.numel() == 0 || out.numel() == 0) {
    return;
  }

  const CTYPE* b_data = self.const_data_ptr<CTYPE>();
  const CTYPE* a_data = mat2.const_data_ptr<CTYPE>();
  CTYPE* c_data = out.mutable_data_ptr<CTYPE>();

  int64_t batch_size = self.size(0);
  int64_t n = self.size(1);
  int64_t k = self.size(2);
  int64_t m = mat2.size(2);

  for (int i = 0; i < batch_size; ++i) {
    const CTYPE* a = a_data + i * m * k;
    const CTYPE* b = b_data + i * k * n;
    CTYPE* c = c_data + i * m * n;

    // clang-format off
    executorch::cpublas::gemm(
        TransposeType::NoTranspose, TransposeType::NoTranspose,
        m, n, k,
        static_cast<CTYPE>(1),
        a, m,
        b, k,
        static_cast<CTYPE>(0),
        c, m);
    // clang-format on
  }
}

Error resize_out_tensor(const Tensor& self, const Tensor& mat2, Tensor& out) {
  executorch::aten::SizesType expected_output_size[kTensorDimensionLimit];

  const size_t m_dim = self.dim() - 2;
  const size_t n_dim = self.dim() - 1;

  for (size_t i = 0; i < m_dim; i++) {
    expected_output_size[i] = self.size(i);
  }

  if (m_dim >= self.dim() || n_dim >= mat2.dim()) {
    ET_LOG(Error, "Incompatible matrix multiply dimensions.");
    return Error::InvalidArgument;
  }

  expected_output_size[m_dim] = self.size(m_dim);
  expected_output_size[n_dim] = mat2.size(n_dim);

  ArrayRef<executorch::aten::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  return resize_tensor(out, output_size);
}
} // namespace

// bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
Tensor& opt_bmm_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      resize_out_tensor(self, mat2, out) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, check_bmm_out_args(self, mat2, out), InvalidArgument, out);

  static constexpr auto name = "bmm.out";

  auto self_type = self.scalar_type();

  if (executorch::runtime::isComplexType(self_type)) {
    ET_SWITCH_COMPLEXH_TYPES(self_type, ctx, name, CTYPE, [&]() {
      bmm_kernel<CTYPE>(self, mat2, out);
    });
  } else {
    ET_SWITCH_REALHBF16_TYPES(self_type, ctx, name, CTYPE, [&]() {
      bmm_kernel<CTYPE>(self, mat2, out);
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
