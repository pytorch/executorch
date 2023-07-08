// Copyright (c) Meta Platforms, Inc. and affiliates.

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

void check_index_args(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& output) {
  // size of indices must not exceed the number of dimensions
  ET_CHECK_MSG(
      indices.size() <= input.dim(),
      "indices.size() %zd > input.dim() %zd",
      ssize_t(indices.size()),
      ssize_t(input.dim()));

  check_indices(input, indices);

  check_index_result_size(input, indices, output);
}

template <typename CTYPE_IN, typename CTYPE_OUT>
void index_out_impl_mask(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  // Data pointers
  const CTYPE_IN* const in_data = input.data_ptr<CTYPE_IN>();
  CTYPE_OUT* const out_data = out.data_ptr<CTYPE_OUT>();

  const Tensor& mask = indices[0].value();
  const bool* const mask_ptr = mask.data_ptr<bool>();
  size_t count = 0;
  for (int i = 0; i < mask.numel(); ++i) {
    if (mask_ptr[i]) {
      out_data[count] = static_cast<CTYPE_OUT>(in_data[i]);
      count++;
    }
  }
}

template <typename CTYPE_IN, typename CTYPE_OUT>
void index_out_impl_list(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  // Data pointers
  const CTYPE_IN* const in_data = input.data_ptr<CTYPE_IN>();
  CTYPE_OUT* dst = out.data_ptr<CTYPE_OUT>();

  size_t num_idx_queries = get_indices_broadcast_len(indices);
  for (size_t idx = 0; idx < num_idx_queries; idx++) {
    const CTYPE_IN* src = in_data;

    // For each index query, align the src and dst pointers to the position
    // described by the query.
    size_t offset = get_index_query_pos_offset(idx, input, indices);
    src += offset;

    // Calculate the region of data to copy for this query.
    // For example, a 2x4x3x5 tensor indexing at [1, 1, :, :] should copy 15
    // elements.
    size_t copy_len = getTrailingDims(input, indices.size() - 1);

    for (size_t i = 0; i < copy_len; ++i) {
      dst[i] = static_cast<CTYPE_OUT>(src[i]);
    }
    dst += copy_len;
  }
}

template <typename CTYPE_IN, typename CTYPE_OUT>
void index_out_impl(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  if (is_index_mask(input, indices)) {
    index_out_impl_mask<CTYPE_IN, CTYPE_OUT>(input, indices, out);
  } else {
    index_out_impl_list<CTYPE_IN, CTYPE_OUT>(input, indices, out);
  }
}

template <typename CTYPE_IN>
inline void index_out_switch_out(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  auto out_type = out.scalar_type();
#define INDEX_COPY_SWITCH_OUTPUT_CASE(ctype, dtype)       \
  case ScalarType::dtype:                                 \
    index_out_impl<CTYPE_IN, ctype>(input, indices, out); \
    break;

  switch (out_type) {
    ET_FORALL_REAL_TYPES_AND(Bool, INDEX_COPY_SWITCH_OUTPUT_CASE);
    default:
      ET_CHECK_MSG(
          false, "%hhd scalar type is not supported for output", out_type);
  }

#undef INDEX_COPY_SWITCH_OUTPUT_CASE
}

inline void index_out_switch_input(
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  auto input_type = input.scalar_type();
#define INDEX_COPY_SWITCH_INPUT_CASE(ctype, dtype)    \
  case ScalarType::dtype:                             \
    index_out_switch_out<ctype>(input, indices, out); \
    break;

  switch (input_type) {
    ET_FORALL_REAL_TYPES_AND(Bool, INDEX_COPY_SWITCH_INPUT_CASE);
    default:
      ET_CHECK_MSG(
          false, "%hhd scalar type is not supported for input", input_type);
  }

#undef INDEX_COPY_SWITCH_INPUT_CASE
}

// expected output dim: 1 + (remaining dimension). Shape: [indices.size,
// *remaining dimension shape]. E.g., 3x3x3x3 tensor, index at [(1, 2), (0,
// 2), :, :] gives output shape [2, 3, 3].
Error resize_out(
    const Tensor& input,
    Tensor& out,
    ArrayRef<exec_aten::optional<Tensor>> indices) {
  size_t out_ndim = 0;
  Tensor::SizesType out_sizes[kTensorDimensionLimit];
  get_index_result_size(input, indices, out_sizes, out_ndim);

  ArrayRef<Tensor::SizesType> output_size{out_sizes, out_ndim};
  auto error = resize_tensor(out, output_size);

  return error;
}
} // namespace

/// aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) ->
/// Tensor(a!)
Tensor& index_Tensor_out(
    RuntimeContext& context,
    const Tensor& input,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  (void)context;

  if (indices.empty()) {
    auto error = resize_tensor(out, input.sizes());
    ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
    memcpy(out.data_ptr<char>(), input.data_ptr<char>(), input.nbytes());
    return out;
  }

  // resize out tensor
  auto error = resize_out(input, out, indices);
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  check_index_args(input, indices, out);

  if (input.numel() == 0) {
    return out;
  }

  index_out_switch_input(input, indices, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
