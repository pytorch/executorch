/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/embedding_bag_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstring>
#include <tuple>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {

// Index tensors may be Int or Long; read and write them as int64_t.
int64_t index_at(const Tensor& t, size_t i) {
  return t.scalar_type() == ScalarType::Long
      ? t.const_data_ptr<int64_t>()[i]
      : static_cast<int64_t>(t.const_data_ptr<int32_t>()[i]);
}

void set_index(Tensor& t, size_t i, int64_t value) {
  if (t.scalar_type() == ScalarType::Long) {
    t.mutable_data_ptr<int64_t>()[i] = value;
  } else {
    t.mutable_data_ptr<int32_t>()[i] = static_cast<int32_t>(value);
  }
}

} // namespace

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
    Tensor& max_indices) {
  // Both only matter to the backward pass.
  (void)scale_grad_by_freq;
  (void)sparse;

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

  const int64_t num_indices = indices.numel();
  const int64_t num_bags = out.size(0);
  const int64_t dim = weight.size(1);

  for (Tensor* t : {&bag_size, &max_indices}) {
    if (t->nbytes() > 0) {
      std::memset(t->mutable_data_ptr(), 0, t->nbytes());
    }
  }

  const auto bag_end = [&](int64_t bag) {
    return bag + 1 < num_bags ? index_at(offsets, bag + 1) : num_indices;
  };
  const auto reduction_end = [&](int64_t bag) {
    return layout.truncate_last_bag ? index_at(offsets, bag + 1) : bag_end(bag);
  };

  // The ATen CPU kernel leaves bag_size at zero for the sum mode, and mirrors
  // it into the 1-D max_indices for the mean mode, where that output carries
  // nothing else.
  if (layout.fills_offset2bag) {
    // With no bags, ATen puts every index before the first bag, in bag -1.
    if (num_bags == 0) {
      for (const auto i : c10::irange(num_indices)) {
        set_index(offset2bag, i, -1);
      }
    }
    for (const auto bag : c10::irange(num_bags)) {
      const int64_t end = bag_end(bag);
      int64_t count = 0;
      for (int64_t i = index_at(offsets, bag); i < end; ++i) {
        set_index(offset2bag, i, bag);
        if (index_at(indices, i) != padding_idx) {
          ++count;
        }
      }
      if (mode != kEmbeddingBagModeSum) {
        set_index(bag_size, bag, count);
      }
      if (mode == kEmbeddingBagModeMean) {
        set_index(max_indices, bag, count);
      }
    }
  }

  if (out.numel() == 0) {
    return ret_val;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "_embedding_bag.out");

  ET_SWITCH_FLOATHBF16_TYPES(weight.scalar_type(), ctx, name, CTYPE, [&]() {
    // Accumulate Half/BFloat16 in float, as ATen does.
    using AccType = std::
        conditional_t<c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>;
    const CTYPE* const weight_data = weight.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
    const CTYPE* const psw_data = per_sample_weights.has_value()
        ? per_sample_weights.value().const_data_ptr<CTYPE>()
        : nullptr;

    const auto weight_strides = weight.strides();
    const auto out_strides = out.strides();
    const auto max_indices_strides = max_indices.strides();

    for (const auto bag : c10::irange(num_bags)) {
      const int64_t start = index_at(offsets, bag);
      const int64_t end = reduction_end(bag);
      // ATen divides by the full bag size even when the fast path truncates
      // the last bag.
      const int64_t mean_divisor =
          mode == kEmbeddingBagModeMean ? index_at(bag_size, bag) : 0;
      for (const auto d : c10::irange(dim)) {
        AccType acc = 0;
        bool seen = false;
        for (int64_t i = start; i < end; ++i) {
          const int64_t row = index_at(indices, i);
          if (row == padding_idx) {
            continue;
          }
          const AccType value = static_cast<AccType>(
              weight_data[row * weight_strides[0] + d * weight_strides[1]]);
          if (mode == kEmbeddingBagModeMax) {
            if (!seen || value > acc) {
              acc = value;
              set_index(
                  max_indices,
                  bag * max_indices_strides[0] + d * max_indices_strides[1],
                  row);
            }
          } else if (psw_data != nullptr) {
            acc += value * static_cast<AccType>(psw_data[i]);
          } else {
            acc += value;
          }
          seen = true;
        }
        if (mean_divisor > 0) {
          acc /= static_cast<AccType>(mean_divisor);
        }
        out_data[bag * out_strides[0] + d * out_strides[1]] =
            static_cast<CTYPE>(acc);
      }
    }
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
