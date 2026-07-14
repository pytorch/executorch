/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <c10/util/irange.h>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

namespace {
using ::at::vec::map;
using ::at::vec::Vectorized;
using ::executorch::aten::Tensor;
using ::executorch::cpublas::gemm;
using ::executorch::cpublas::TransposeType;
using ::executorch::runtime::toString;

// Use vector store to initialize with scalar bias.
template <typename scalar_t>
void initialize_scalar(
    const ssize_t out_numel,
    const scalar_t init,
    scalar_t* out) {
  using Vec = Vectorized<scalar_t>;

  // Initialize a vector with the scalar initial value.
  Vec init_vec(init);

  ssize_t d = 0;
  for (; d < out_numel - (out_numel % Vec::size()); d += Vec::size()) {
    // Vector-length store.
    init_vec.store(out + d);
  }
  if (out_numel - d > 0) {
    // Sub-vector-length store.
    init_vec.store(out + d, static_cast<int>(out_numel - d));
  }
}

// Use std::memcpy to initialize with vector bias.
template <typename scalar_t>
void initialize_to_vector(
    const ssize_t n,
    const ssize_t m,
    const scalar_t* bias,
    scalar_t* out) {
  // Output is a n x m x scalar_t, while bias is m x scalar_t.
  const size_t row_size = static_cast<size_t>(m) * sizeof(scalar_t);
  for (const auto col : c10::irange(n)) {
    std::memcpy(
        // Point to Column `col` of the output tensor.
        out + col * m,
        bias,
        row_size);
  }
}

} // namespace

Tensor& opt_linear_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_linear_args(in, mat2, out), InvalidArgument, out);

  size_t output_ndim = 0;
  std::array<executorch::aten::SizesType, kTensorDimensionLimit> output_sizes;
  get_linear_out_target_size(in, mat2, output_sizes.data(), &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes.data(), output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  // gemm on some platforms doesn't tolerate empty input.
  if (out.numel() == 0) {
    return out;
  }

  ssize_t n = 1;
  for (int ii = 0; ii < in.dim() - 1; ++ii) {
    n *= in.sizes()[ii];
  }
  const ssize_t k = in.sizes()[in.dim() - 1];
  const ssize_t m = mat2.size(0);

  if (bias.has_value()) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        // Bias and output dtype must match.
        bias->dtype() == out.dtype(),
        InvalidArgument,
        out,
        "Bias has wrong dtype! Expected bias dtype to be the same as out dtype %s"
        " but got %s",
        toString(bias->dtype()),
        toString(out.dtype()));

    ET_KERNEL_CHECK_MSG(
        ctx,
        // Either no bias or bias is a 1D tensor of size m or 1.
        bias->dim() == 1 && (bias->size(0) == m || bias->size(0) == 1),
        InvalidArgument,
        out,
        "Bias has wrong dimensionality! Expected 1-D tensor of size %d or empty,"
        " but got %d-D tensor with %d elements",
        static_cast<int>(m),
        static_cast<int>(bias->dim()),
        static_cast<int>(bias->numel()));
  }

  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, out.scalar_type(), ctx, "linear.out", CTYPE, [&] {
        // Fill output with bias if it is provided.
        if (bias.has_value() && bias->numel() == 1) {
          // Scalar version of initialization.
          initialize_scalar<CTYPE>(
              out.numel(),
              *bias->const_data_ptr<CTYPE>(),
              out.mutable_data_ptr<CTYPE>());
        } else if (bias.has_value()) {
          // Assume bias is a 1D tensor of size m.
          initialize_to_vector<CTYPE>(
              n,
              m,
              bias->const_data_ptr<CTYPE>(),
              out.mutable_data_ptr<CTYPE>());
        }

        // Set beta to 1 if bias was applied so that GEMM adds to the pre-filled
        // bias, otherwise beta remains 0 (i.e. the output is fully overwritten
        // by GEMM).
        const CTYPE beta =
            bias.has_value() ? static_cast<CTYPE>(1) : static_cast<CTYPE>(0);

        gemm(
            /*transa=*/TransposeType::Transpose,
            /*transb=*/TransposeType::NoTranspose,
            m,
            n,
            k,
            /*alpha=*/static_cast<CTYPE>(1),
            mat2.const_data_ptr<CTYPE>(),
            k,
            in.const_data_ptr<CTYPE>(),
            k,
            beta,
            out.mutable_data_ptr<CTYPE>(),
            m);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
