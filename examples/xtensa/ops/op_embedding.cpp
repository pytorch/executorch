/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

void embedding_out(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse,
    Tensor& out) {
  int64_t nbytes_per_entry = weight.size(1) * weight.element_size();
  const char* w_data = weight.const_data_ptr<char>();
  char* out_data = out.mutable_data_ptr<char>();
  const int64_t* indices_ptr = indices.const_data_ptr<int64_t>();

  for (int i = 0, e = indices.numel(); i < e; i++) {
    // memcpy(dest, src, nbytes);
    memcpy(
        out_data, w_data + nbytes_per_entry * indices_ptr[i], nbytes_per_entry);
    out_data += nbytes_per_entry;
  }
}

} // namespace native
} // namespace executor
} // namespace torch
