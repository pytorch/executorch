/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/kernels/portable/cpu/util/index_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

void check_index_put_args(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    Tensor& output) {
  // size of indices must not exceed the number of dimensions
  ET_CHECK_MSG(
      indices.size() <= input.dim(),
      "indices.size() %zd > input.dim() %zd",
      ssize_t(indices.size()),
      ssize_t(input.dim()));

  // input and output should have the same dtype
  ET_CHECK_SAME_DTYPE3(input, output, values);

  check_indices(input, indices);

  // If values not broadcastable, then check it is equal to the size of the
  // indexing result.
  if (values.numel() != 1) {
    check_index_result_size(input, indices, values);
  }
}

template <typename CTYPE>
void index_put_out_impl_mask(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  // Data pointers
  const CTYPE* const in_data = input.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const CTYPE* val_data = values.const_data_ptr<CTYPE>();

  // To start, copy the input into the output
  memcpy(out_data, in_data, input.nbytes());

  const Tensor& mask = indices[0].value();
  const bool* const mask_ptr = mask.const_data_ptr<bool>();
  size_t count = 0;
  for (int i = 0; i < mask.numel(); ++i) {
    if (mask_ptr[i]) {
      if (accum) {
        out_data[i] += val_data[count];
      } else {
        out_data[i] = val_data[count];
      }
      if (values.numel() > 1) {
        count++;
      }
    }
  }
}

template <typename CTYPE>
void index_put_out_impl_list(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  // Data pointers
  const CTYPE* const in_data = input.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const CTYPE* val = values.const_data_ptr<CTYPE>();

  // To start, copy the input into the output
  memcpy(out_data, in_data, input.nbytes());

  size_t num_idx_queries = get_indices_broadcast_len(indices);
  for (size_t idx = 0; idx < num_idx_queries; idx++) {
    const CTYPE* src = in_data;
    CTYPE* dst = out_data;

    // For each index query, align the src and dst pointers to the position
    // described by the query.
    size_t offset = get_index_query_pos_offset(idx, input, indices);
    src += offset;
    dst += offset;

    // Calculate the region of data to copy for this query.
    // For example, a 2x4x3x5 tensor indexing at [1, 1, :, :] should copy 15
    // elements.
    size_t copy_len = getTrailingDims(input, indices.size() - 1);

    // If values only contains 1 element, it needs to be broadcasted.
    if (values.numel() == 1) {
      CTYPE value = *val;

      for (size_t i = 0; i < copy_len; ++i) {
        if (accum) {
          dst[i] += value;
        } else {
          dst[i] = value;
        }
      }
    }
    // General case.
    else {
      if (accum) {
        for (size_t i = 0; i < copy_len; ++i) {
          dst[i] = src[i] + val[i];
        }
        val += copy_len;
      } else {
        size_t copy_size = copy_len * sizeof(CTYPE);
        memcpy(dst, val, copy_size);
        val += copy_len;
      }
    }
  }
}

template <typename CTYPE>
void index_put_out_impl(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  if (is_index_mask(input, indices)) {
    index_put_out_impl_mask<CTYPE>(input, indices, values, accum, out);
  } else {
    index_put_out_impl_list<CTYPE>(input, indices, values, accum, out);
  }
}

inline void index_put_out_switch_input(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  auto input_type = input.scalar_type();
#define INDEX_PUT_SWITCH_INPUT_CASE(ctype, dtype)                  \
  case ScalarType::dtype:                                          \
    index_put_out_impl<ctype>(input, indices, values, accum, out); \
    break;

  switch (input_type) {
    ET_FORALL_REAL_TYPES_AND(Bool, INDEX_PUT_SWITCH_INPUT_CASE);
    default:
      ET_CHECK_MSG(
          false, "%hhd scalar type is not supported for input", input_type);
  }

#undef INDEX_PUT_SWITCH_INPUT_CASE
}

// Output tensor should be the same size as the input tensor
Error resize_like_input(const Tensor& input, Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < input.dim(); ++i) {
    expected_output_size[i] = input.size(i);
  }
  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(input.dim())};
  auto error = resize_tensor(out, output_size);

  return error;
}

} // namespace

/// aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values,
/// bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor& index_put_out(
    RuntimeContext& context,
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accumulate,
    Tensor& out) {
  (void)context;

  if (indices.empty()) {
    auto error = resize_tensor(out, input.sizes());
    ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
    memcpy(
        out.mutable_data_ptr<char>(),
        input.const_data_ptr<char>(),
        input.nbytes());
    return out;
  }

  // resize out tensor
  auto error = resize_like_input(input, out);
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  check_index_put_args(input, indices, values, out);

  if (input.numel() == 0) {
    return out;
  }

  index_put_out_switch_input(input, indices, values, accumulate, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
