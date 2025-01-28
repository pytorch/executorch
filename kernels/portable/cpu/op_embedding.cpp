/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

// A simple lookup table that looks up embeddings in a fixed dictionary and
// size.

// This module is often used to retrieve word embeddings using indices. The
// input to the module is a list of indices, and the embedding matrix, and the
// output is the corresponding word embeddings.
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE>
void embedding_kernel(
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& indices,
    Tensor& out) {
  int64_t nbytes_per_entry = weight.size(1) * weight.element_size();
  const char* w_data = weight.const_data_ptr<char>();
  char* out_data = out.mutable_data_ptr<char>();
  const CTYPE* indices_ptr = indices.const_data_ptr<CTYPE>();
  ssize_t weight_height = weight.size(0);
  const auto indices_numel = indices.numel();
  for (int i = 0; i < indices_numel; i++) {
    // Ensure index is larger than 0 and smaller than weight.size(0)
    ET_KERNEL_CHECK_MSG(
        ctx,
        indices_ptr[i] < weight_height,
        InvalidArgument,
        ,
        "indices_ptr[%d] %ld >= weight.size(0) %zd",
        i,
        static_cast<long>(indices_ptr[i]),
        weight_height);
    ET_KERNEL_CHECK_MSG(
        ctx,
        indices_ptr[i] >= 0,
        InvalidArgument,
        ,
        "indices_ptr[%d] %ld < 0",
        i,
        static_cast<long>(indices_ptr[i]));
    if (w_data != nullptr) {
      memcpy(
          out_data,
          w_data + nbytes_per_entry * indices_ptr[i],
          nbytes_per_entry);
    }
    out_data += nbytes_per_entry;
  }
}
} // namespace

// embedding.out(Tensor weight, Tensor indices, int padding_idx=-1, bool
// scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor& embedding_out(
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse,
    Tensor& out) {
  (void)ctx;
  (void)padding_idx;
  (void)scale_grad_by_freq;
  (void)sparse;

  ET_KERNEL_CHECK(
      ctx, check_embedding_args(weight, indices, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_embedding_output(weight, indices, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.size(out.dim() - 1) == weight.size(1),
      InvalidArgument,
      out,
      "out.size(%zd) %zd != weight.size(1) %zd",
      out.dim() - 1,
      out.size(1),
      weight.size(1));

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(weight, indices, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(weight), InvalidArgument, out);

  ScalarType ix_type = indices.scalar_type();
  ET_CHECK_MSG(
      ix_type == ScalarType::Long || ix_type == ScalarType::Int,
      "Expected indices tensor to have Long or Int scalar types");

  ET_SWITCH_TWO_TYPES(
      Long, Int, ix_type, ctx, "op_embedding.out", CTYPE, [&]() {
        embedding_kernel<CTYPE>(ctx, weight, indices, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
