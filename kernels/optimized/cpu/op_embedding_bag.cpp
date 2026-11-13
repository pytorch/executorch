/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/embedding_bag_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <algorithm>
#include <tuple>

namespace torch::executor::native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {

using Vec = at::vec::Vectorized<float>;

// Export allocates offset2bag, bag_size and max_indices with the offsets
// dtype, even though eager ATen promotes them to the wider index dtype.
bool is_fast_path(
    const Tensor& weight,
    const Tensor& offsets,
    bool include_last_offset,
    const Tensor& out,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& max_indices) {
  return weight.scalar_type() == ScalarType::Float &&
      tensor_is_default_dim_order(weight) && tensor_is_default_dim_order(out) &&
      tensor_is_default_dim_order(max_indices) &&
      offsets.numel() > static_cast<ssize_t>(include_last_offset) &&
      offset2bag.scalar_type() == offsets.scalar_type() &&
      bag_size.scalar_type() == offsets.scalar_type() &&
      max_indices.scalar_type() == offsets.scalar_type();
}

// Requires check_embedding_bag_offsets_and_indices and
// resize_embedding_bag_outputs; every bag writes only its own output rows and
// offset2bag range.
template <typename CTYPE_IX, typename CTYPE_OFF>
void embedding_bag_kernel(
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    int64_t padding_idx,
    const EmbeddingBagLayout& layout,
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices) {
  const CTYPE_IX* indices_data = indices.const_data_ptr<CTYPE_IX>();
  const CTYPE_OFF* offsets_data = offsets.const_data_ptr<CTYPE_OFF>();
  const int64_t num_indices = indices.numel();
  const int64_t num_bags = out.size(0);
  const int64_t width = weight.size(1);
  // Partial at::vec loads and stores go through a stack buffer on NEON.
  const int64_t vec_width = width - width % Vec::size();
  const bool fills_offset2bag = layout.fills_offset2bag;
  const bool truncate_last_bag = layout.truncate_last_bag;

  const float* weight_data = weight.const_data_ptr<float>();
  const float* scales = per_sample_weights.has_value()
      ? per_sample_weights->const_data_ptr<float>()
      : nullptr;
  float* out_data = out.mutable_data_ptr<float>();
  CTYPE_OFF* offset2bag_data = offset2bag.mutable_data_ptr<CTYPE_OFF>();
  CTYPE_OFF* bag_size_data = bag_size.mutable_data_ptr<CTYPE_OFF>();
  CTYPE_OFF* max_indices_data = max_indices.mutable_data_ptr<CTYPE_OFF>();

  // Aim for about GRAIN_SIZE weight reads per task.
  const int64_t reads_per_bag =
      std::max<int64_t>(1, num_indices * width / num_bags);
  const bool success = ::executorch::extension::parallel_for(
      0,
      num_bags,
      std::max<int64_t>(
          1, ::executorch::extension::internal::GRAIN_SIZE / reads_per_bag),
      [&](const int64_t bags_begin, const int64_t bags_end) {
        for (const auto bag : c10::irange(bags_begin, bags_end)) {
          const int64_t begin = offsets_data[bag];
          const int64_t end =
              bag + 1 < num_bags ? offsets_data[bag + 1] : num_indices;
          // truncate_last_bag implies include_last_offset, so
          // offsets[bag + 1] exists for every bag.
          const int64_t reduce_end =
              truncate_last_bag ? offsets_data[bag + 1] : end;
          float* out_row = out_data + bag * width;
          std::fill_n(out_row, width, 0.0f);
          CTYPE_OFF* winners = nullptr;
          if (mode == kEmbeddingBagModeMax) {
            winners = max_indices_data + bag * width;
            std::fill_n(winners, width, CTYPE_OFF(0));
          }
          CTYPE_OFF valid = 0;
          for (int64_t i = begin; i < end; ++i) {
            if (fills_offset2bag) {
              offset2bag_data[i] = static_cast<CTYPE_OFF>(bag);
            }
            const int64_t index = indices_data[i];
            if (index == padding_idx) {
              continue;
            }
            ++valid;
            if (i >= reduce_end) {
              continue;
            }
            const float* row = weight_data + index * width;
            if (mode == kEmbeddingBagModeMax) {
              for (const auto d : c10::irange(width)) {
                if (valid == 1 || row[d] > out_row[d]) {
                  out_row[d] = row[d];
                  winners[d] = static_cast<CTYPE_OFF>(index);
                }
              }
            } else if (scales != nullptr) {
              const float scale = scales[i];
              const Vec scale_vec(scale);
              at::vec::map2<float>(
                  [scale_vec](Vec x, Vec y) {
                    return at::vec::fmadd(y, scale_vec, x);
                  },
                  out_row,
                  out_row,
                  row,
                  vec_width);
              for (const auto d : c10::irange(vec_width, width)) {
                out_row[d] += row[d] * scale;
              }
            } else {
              at::vec::map2<float>(
                  [](Vec x, Vec y) { return x + y; },
                  out_row,
                  out_row,
                  row,
                  vec_width);
              for (const auto d : c10::irange(vec_width, width)) {
                out_row[d] += row[d];
              }
            }
          }
          bag_size_data[bag] =
              mode == kEmbeddingBagModeSum ? CTYPE_OFF(0) : valid;
          if (mode == kEmbeddingBagModeSum) {
            max_indices_data[bag] = CTYPE_OFF(0);
          } else if (mode == kEmbeddingBagModeMean) {
            max_indices_data[bag] = valid;
            if (valid != 0) {
              const float divisor = static_cast<float>(valid);
              const Vec divisor_vec(divisor);
              at::vec::map<float>(
                  [divisor_vec](Vec x) { return x / divisor_vec; },
                  out_row,
                  out_row,
                  vec_width);
              for (const auto d : c10::irange(vec_width, width)) {
                out_row[d] /= divisor;
              }
            }
          }
        }
      });
  ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
}

} // namespace

// Portable kernel (same-op fallback). Both libs link into the same binary.
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _embedding_bag_out(
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices);

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> opt_embedding_bag_out(
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices) {
  if (!is_fast_path(
          weight,
          offsets,
          include_last_offset,
          out,
          offset2bag,
          bag_size,
          max_indices)) {
    return _embedding_bag_out(
        ctx,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        out,
        offset2bag,
        bag_size,
        max_indices);
  }

  std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> ret_val(
      out, offset2bag, bag_size, max_indices);

  ET_KERNEL_CHECK(
      ctx,
      check_embedding_bag_args(
          weight,
          indices,
          offsets,
          mode,
          per_sample_weights,
          include_last_offset,
          out,
          offset2bag,
          bag_size,
          max_indices),
      InvalidArgument,
      ret_val);
  ET_KERNEL_CHECK(
      ctx,
      check_embedding_bag_offsets_and_indices(weight, indices, offsets),
      InvalidArgument,
      ret_val);

  EmbeddingBagLayout layout{};
  ET_KERNEL_CHECK(
      ctx,
      resize_embedding_bag_outputs(
          weight,
          indices,
          offsets,
          mode,
          per_sample_weights,
          include_last_offset,
          padding_idx,
          out,
          offset2bag,
          bag_size,
          max_indices,
          layout) == Error::Ok,
      InvalidArgument,
      ret_val);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "_embedding_bag.out";

  ET_SWITCH_TWO_TYPES(
      Long, Int, indices.scalar_type(), ctx, op_name, CTYPE_IX, [&]() {
        ET_SWITCH_TWO_TYPES(
            Long, Int, offsets.scalar_type(), ctx, op_name, CTYPE_OFF, [&]() {
              embedding_bag_kernel<CTYPE_IX, CTYPE_OFF>(
                  ctx,
                  weight,
                  indices,
                  offsets,
                  mode,
                  per_sample_weights,
                  padding_idx,
                  layout,
                  out,
                  offset2bag,
                  bag_size,
                  max_indices);
            });
      });

  return ret_val;
}

} // namespace torch::executor::native
