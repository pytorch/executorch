/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <c10/util/irange.h>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_padding_args(
    int64_t n,
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> padding,
    Tensor& out,
    bool reflection = false);

void get_padding_out_target_size(
    int64_t n,
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> padding,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

inline int64_t replication_ix(int64_t j, int64_t size, int64_t pad) {
  return j < pad ? 0 : j >= pad && j < size + pad ? j - pad : size - 1;
}

inline int64_t reflection_ix(int64_t j, int64_t size, int64_t pad) {
  return j < pad                   ? pad - j
      : j >= pad && j < size + pad ? j - pad
                                   : 2 * size + pad - j - 2;
}

template <typename CTYPE, typename PaddingIx>
void pad1d(
    const PaddingIx& padding_ix,
    const Tensor& in,
    Tensor& out,
    executorch::aten::ArrayRef<int64_t> padding) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const auto dim = in.dim() - 1;
  const auto outer = getLeadingDims(out, dim);
  const auto in_width = in.size(dim);
  const auto out_width = out.size(dim);
  const auto pad_left = padding[0];
  for (const auto i : c10::irange(outer)) {
    size_t out_i_base = i * out_width;
    size_t in_i_base = i * in_width;
    for (const auto w : c10::irange(out_width)) {
      int64_t in_w_idx = padding_ix(w, in_width, pad_left);
      ET_CHECK(in_w_idx >= 0 && in_w_idx < in_width);
      out_data[out_i_base + w] = in_data[in_i_base + in_w_idx];
    }
  }
}

template <typename CTYPE, typename PaddingIx>
void pad2d(
    const PaddingIx& padding_ix,
    const Tensor& in,
    Tensor& out,
    executorch::aten::ArrayRef<int64_t> padding) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const auto dim = in.dim() - 2;
  const auto outer = getLeadingDims(out, dim);
  const auto in_height = in.size(dim);
  const auto in_width = in.size(dim + 1);
  const auto out_height = out.size(dim);
  const auto out_width = out.size(dim + 1);
  const auto pad_left = padding[0];
  const auto pad_top = padding[2];

  for (const auto i : c10::irange(outer)) {
    size_t out_i_base = i * out_height * out_width;
    size_t in_i_base = i * in_height * in_width;
    for (const auto h : c10::irange(out_height)) {
      size_t out_h_base = out_i_base + h * out_width;
      int64_t in_h_idx = padding_ix(h, in_height, pad_top);
      ET_CHECK(in_h_idx >= 0 && in_h_idx < in_height);
      size_t in_h_base = in_i_base + in_h_idx * in_width;
      for (const auto w : c10::irange(out_width)) {
        int64_t in_w_idx = padding_ix(w, in_width, pad_left);
        ET_CHECK(in_w_idx >= 0 && in_w_idx < in_width);
        out_data[out_h_base + w] = in_data[in_h_base + in_w_idx];
      }
    }
  }
}

template <typename CTYPE, typename PaddingIx>
void pad3d(
    const PaddingIx& padding_ix,
    const Tensor& in,
    Tensor& out,
    executorch::aten::ArrayRef<int64_t> padding) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const auto dim = in.dim() - 3;
  const auto outer = getLeadingDims(out, dim);
  const auto in_depth = in.size(dim);
  const auto in_height = in.size(dim + 1);
  const auto in_width = in.size(dim + 2);
  const auto out_depth = out.size(dim);
  const auto out_height = out.size(dim + 1);
  const auto out_width = out.size(dim + 2);
  const auto pad_left = padding[0];
  const auto pad_top = padding[2];
  const auto pad_front = padding[4];

  for (const auto i : c10::irange(outer)) {
    size_t out_i_base = i * out_depth * out_height * out_width;
    size_t in_i_base = i * in_depth * in_height * in_width;
    for (const auto d : c10::irange(out_depth)) {
      size_t out_d_base = out_i_base + d * out_height * out_width;
      int64_t in_d_base_padding = padding_ix(d, in_depth, pad_front);
      ET_CHECK(in_d_base_padding >= 0 && in_d_base_padding < in_depth);
      size_t in_d_base = in_i_base + in_d_base_padding * in_height * in_width;
      for (const auto h : c10::irange(out_height)) {
        size_t out_h_base = out_d_base + h * out_width;
        int64_t in_h_base_padding = padding_ix(h, in_height, pad_top);
        ET_CHECK(in_h_base_padding >= 0 && in_h_base_padding < in_height);
        size_t in_h_base = in_d_base + in_h_base_padding * in_width;
        for (const auto w : c10::irange(out_width)) {
          int64_t in_w_base_padding = padding_ix(w, in_width, pad_left);
          ET_CHECK(in_w_base_padding >= 0 && in_w_base_padding < in_width);
          out_data[out_h_base + w] = in_data[in_h_base + in_w_base_padding];
        }
      }
    }
  }
}

} // namespace executor
} // namespace torch
