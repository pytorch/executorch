/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

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
    const Tensor& weight,
    const Tensor& indices,
    Tensor& out) {
  int64_t nbytes_per_entry = weight.size(1) * weight.element_size();
  const char* w_data = weight.const_data_ptr<char>();
  char* out_data = out.mutable_data_ptr<char>();
  const CTYPE* indices_ptr = indices.const_data_ptr<CTYPE>();
  ssize_t weight_height = weight.size(0);
  for (int i = 0; i < indices.numel(); i++) {
    // Ensure index is larger than 0 and smaller than weight.size(0)
    ET_CHECK_MSG(
        indices_ptr[i] < weight_height,
        "indices_ptr[%d] %ld >= weight.size(0) %zd",
        i,
        static_cast<long>(indices_ptr[i]),
        weight_height);
    ET_CHECK_MSG(
        indices_ptr[i] >= 0,
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

void resize_out_tensor(
    const Tensor& weight,
    const Tensor& indices,
    Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < indices.dim(); i++) {
    expected_output_size[i] = indices.size(i);
  }
  const size_t embedding_dim = weight.size(1);
  expected_output_size[out.dim() - 1] = embedding_dim;

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in embedding_out");
}
} // namespace

// embedding.out(Tensor weight, Tensor indices, int padding_idx=-1, bool
// scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor& embedding_out(
    RuntimeContext& ctx,
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

  // Ensure weight is 2-D. It could be empty.
  ET_CHECK_MSG(weight.dim() == 2, "weight.dim() %zd != 2", weight.dim());

  // Ensure out is k+1 dimension tensor where k is the indices.dim()
  // out's first k dimension shall be same as indices, and the last dim shall
  // equal weight's last dim
  ET_CHECK_MSG(
      out.dim() == indices.dim() + 1,
      "out.dim() %zd != indices.dim() %zd + 1",
      out.dim(),
      indices.dim());

  resize_out_tensor(weight, indices, out);

  for (size_t i = 0; i < indices.dim(); i++) {
    ET_CHECK_MSG(
        out.size(i) == indices.size(i),
        "out.size(%zd) %zd != indices.size(%zd) %zd",
        i,
        out.size(i),
        i,
        indices.size(i));
  }
  ET_CHECK_MSG(
      out.size(out.dim() - 1) == weight.size(1),
      "out.size(%zd) %zd != weight.size(1) %zd",
      out.dim() - 1,
      out.size(1),
      weight.size(1));

  // Ensure dtype is the same for out and weight
  ET_CHECK_SAME_DTYPE2(weight, out);

  ScalarType ix_type = indices.scalar_type();
  ET_CHECK_MSG(
      ix_type == ScalarType::Long || ix_type == ScalarType::Int,
      "Expected indices tensor to have Long or Int scalar types");

  ET_SWITCH_TWO_TYPES(
      Long, Int, ix_type, ctx, "op_embedding.out", CTYPE, [&]() {
        embedding_kernel<CTYPE>(weight, indices, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
