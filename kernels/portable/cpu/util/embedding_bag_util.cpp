/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/embedding_bag_util.h>

#include <c10/util/irange.h>

#include <cinttypes>

namespace torch {
namespace executor {

namespace {

using SizesType = executorch::aten::SizesType;

bool is_index_dtype(ScalarType dtype) {
  return dtype == ScalarType::Long || dtype == ScalarType::Int;
}

template <typename CTYPE>
bool check_offsets(const CTYPE* offsets, int64_t count, int64_t num_indices) {
  int64_t prev_offset = 0;
  for (const auto i : c10::irange(count)) {
    const int64_t offset = offsets[i];
    ET_CHECK_OR_RETURN_FALSE(
        (i > 0 || offset == 0) && offset >= prev_offset &&
            offset <= num_indices,
        "offsets must start at 0 and be nondecreasing up to %" PRId64
        "; offsets[%zu] = %" PRId64,
        num_indices,
        static_cast<size_t>(i),
        offset);
    prev_offset = offset;
  }
  return true;
}

template <typename CTYPE>
bool check_indices(const CTYPE* indices, int64_t count, int64_t num_rows) {
  for (const auto i : c10::irange(count)) {
    const int64_t row = indices[i];
    ET_CHECK_OR_RETURN_FALSE(
        row >= 0 && row < num_rows,
        "index %" PRId64 " is outside weight rows [0, %" PRId64 ")",
        row,
        num_rows);
  }
  return true;
}

// Mirrors ATen's is_fast_path in EmbeddingBag.cpp.
bool is_aten_fast_path(
    const Tensor& weight,
    const std::optional<Tensor>& per_sample_weights,
    const Tensor& out,
    int64_t padding_idx) {
  const auto dtype = weight.scalar_type();
  return (dtype == ScalarType::Float || dtype == ScalarType::Half ||
          dtype == ScalarType::BFloat16) &&
      weight.strides()[1] == 1 && out.strides()[1] == 1 && padding_idx < 0 &&
      (!per_sample_weights.has_value() ||
       per_sample_weights->strides()[0] == 1);
}

} // namespace

bool check_embedding_bag_args(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    const Tensor& out,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& max_indices) {
  ET_CHECK_OR_RETURN_FALSE(
      weight.dim() == 2,
      "weight must be 2-D; weight.dim() = %" ET_PRI_TENSOR_DIM,
      weight.dim());
  ET_CHECK_OR_RETURN_FALSE(
      indices.dim() == 1,
      "indices must be 1-D; indices.dim() = %" ET_PRI_TENSOR_DIM,
      indices.dim());
  ET_CHECK_OR_RETURN_FALSE(
      offsets.dim() == 1,
      "offsets must be 1-D; offsets.dim() = %" ET_PRI_TENSOR_DIM,
      offsets.dim());
  ET_CHECK_OR_RETURN_FALSE(
      is_index_dtype(indices.scalar_type()),
      "indices must be Long or Int; indices.scalar_type() = %s",
      toString(indices.scalar_type()));
  ET_CHECK_OR_RETURN_FALSE(
      is_index_dtype(offsets.scalar_type()),
      "offsets must be Long or Int; offsets.scalar_type() = %s",
      toString(offsets.scalar_type()));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(weight));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(weight, out));

  // Eager ATen gives these the promoted indices/offsets dtype, but export
  // plans them with meta_embedding_bag, which uses the offsets dtype.
  for (const Tensor* t : {&offset2bag, &bag_size, &max_indices}) {
    ET_CHECK_OR_RETURN_FALSE(
        is_index_dtype(t->scalar_type()),
        "index outputs must be Long or Int; got %s",
        toString(t->scalar_type()));
  }

  ET_CHECK_OR_RETURN_FALSE(
      mode == kEmbeddingBagModeSum || mode == kEmbeddingBagModeMean ||
          mode == kEmbeddingBagModeMax,
      "mode must be 0 (sum), 1 (mean) or 2 (max); mode = %" PRId64,
      mode);

  if (include_last_offset) {
    ET_CHECK_OR_RETURN_FALSE(
        offsets.numel() >= 1,
        "offsets must hold at least the final offset when include_last_offset is set");
  }

  if (per_sample_weights.has_value()) {
    const Tensor& psw = per_sample_weights.value();
    ET_CHECK_OR_RETURN_FALSE(psw.dim() == 1, "per_sample_weights must be 1-D");
    ET_CHECK_OR_RETURN_FALSE(
        mode == kEmbeddingBagModeSum,
        "per_sample_weights is only supported for mode 0 (sum); mode = %" PRId64,
        mode);
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(weight, psw));
    ET_CHECK_OR_RETURN_FALSE(
        psw.numel() == indices.numel(),
        "per_sample_weights must have one entry per index; got %zd for %zd indices",
        psw.numel(),
        indices.numel());
  }

  return true;
}

bool check_embedding_bag_offsets_and_indices(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets) {
  const int64_t num_indices = indices.numel();
  const bool offsets_ok = offsets.scalar_type() == ScalarType::Long
      ? check_offsets(
            offsets.const_data_ptr<int64_t>(), offsets.numel(), num_indices)
      : check_offsets(
            offsets.const_data_ptr<int32_t>(), offsets.numel(), num_indices);
  if (!offsets_ok) {
    return false;
  }
  // Validate every index, including those past a partial final offset.
  const int64_t num_rows = weight.size(0);
  return indices.scalar_type() == ScalarType::Long
      ? check_indices(indices.const_data_ptr<int64_t>(), num_indices, num_rows)
      : check_indices(indices.const_data_ptr<int32_t>(), num_indices, num_rows);
}

Error resize_embedding_bag_outputs(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    EmbeddingBagLayout& layout) {
  const int64_t num_indices = indices.numel();
  const SizesType out_sizes[2] = {
      static_cast<SizesType>(
          offsets.numel() - static_cast<int64_t>(include_last_offset)),
      static_cast<SizesType>(weight.size(1))};
  // out must be resized first: the fast-path check reads its strides.
  ET_CHECK_OK_OR_RETURN_ERROR(resize_tensor(out, {out_sizes, 2}));

  // Kernels may use one reduction loop for both ATen paths, but follow
  // ATen's path choice so the auxiliary outputs and the partial final bag
  // match, including statically allocated export outputs.
  const bool fast_path =
      is_aten_fast_path(weight, per_sample_weights, out, padding_idx);
  // ATen skips offset2bag only for sum on its fast path, but export sizes it
  // with meta_embedding_bag, whose fast path differs (e.g. for BFloat16). Sum
  // never reads it, so keep a caller-allocated num_indices layout and fill it.
  layout.fills_offset2bag = mode != kEmbeddingBagModeSum || !fast_path ||
      offset2bag.numel() == num_indices;
  layout.truncate_last_bag =
      include_last_offset && fast_path && mode != kEmbeddingBagModeMax;

  // The auxiliary outputs exist for the backward pass, and the ATen CPU kernel
  // only fills the ones each mode needs; match it so shapes agree.
  // bag_size and the non-max max_indices are {num_bags}, a prefix of
  // out_sizes.
  const SizesType offset2bag_size =
      static_cast<SizesType>(layout.fills_offset2bag ? num_indices : 0);
  ET_CHECK_OK_OR_RETURN_ERROR(resize_tensor(offset2bag, {&offset2bag_size, 1}));
  ET_CHECK_OK_OR_RETURN_ERROR(resize_tensor(bag_size, {out_sizes, 1}));
  return resize_tensor(
      max_indices, {out_sizes, mode == kEmbeddingBagModeMax ? 2u : 1u});
}

} // namespace executor
} // namespace torch
