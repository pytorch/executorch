/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;

namespace {

// aten::_embedding_bag's `mode` argument.
constexpr int64_t kModeSum = 0;
constexpr int64_t kModeMean = 1;
constexpr int64_t kModeMax = 2;

bool is_index_dtype(ScalarType dtype) {
  return dtype == ScalarType::Long || dtype == ScalarType::Int;
}

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

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(weight, out));
  const ScalarType index_type = indices.scalar_type() == ScalarType::Long ||
          offsets.scalar_type() == ScalarType::Long
      ? ScalarType::Long
      : ScalarType::Int;
  for (const Tensor* t : {&offset2bag, &bag_size, &max_indices}) {
    ET_CHECK_OR_RETURN_FALSE(
        t->scalar_type() == index_type,
        "index outputs must have promoted indices/offsets dtype %s, but got %s",
        toString(index_type),
        toString(t->scalar_type()));
  }

  ET_CHECK_OR_RETURN_FALSE(
      mode == kModeSum || mode == kModeMean || mode == kModeMax,
      "mode must be 0 (sum), 1 (mean) or 2 (max); mode = %" PRId64,
      mode);

  if (include_last_offset) {
    ET_CHECK_OR_RETURN_FALSE(
        offsets.numel() >= 1,
        "offsets must hold at least the final offset when include_last_offset is set");
  }

  if (per_sample_weights.has_value()) {
    ET_CHECK_OR_RETURN_FALSE(
        per_sample_weights.value().dim() == 1,
        "per_sample_weights must be 1-D");
    ET_CHECK_OR_RETURN_FALSE(
        mode == kModeSum,
        "per_sample_weights is only supported for mode 0 (sum); mode = %" PRId64,
        mode);
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_dtype(weight, per_sample_weights.value()));
    ET_CHECK_OR_RETURN_FALSE(
        per_sample_weights.value().numel() == indices.numel(),
        "per_sample_weights must have one entry per index; got %zd for %zd indices",
        per_sample_weights.value().numel(),
        indices.numel());
  }

  return true;
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

  const int64_t num_indices = indices.numel();
  const int64_t num_bags =
      include_last_offset ? offsets.numel() - 1 : offsets.numel();
  const int64_t dim = weight.size(1);

  for (const auto i : c10::irange(offsets.numel())) {
    const int64_t offset = index_at(offsets, i);
    ET_KERNEL_CHECK_MSG(
        ctx,
        offset >= 0 && offset <= num_indices &&
            (i == 0 ? offset == 0 : offset >= index_at(offsets, i - 1)),
        InvalidArgument,
        ret_val,
        "offsets must start at zero and be nondecreasing within indices length");
  }
  // Validate even unused indices and zero-width outputs before touching data.
  for (const auto i : c10::irange(num_indices)) {
    const int64_t row = index_at(indices, i);
    ET_KERNEL_CHECK_MSG(
        ctx,
        row >= 0 && row < weight.size(0),
        InvalidArgument,
        ret_val,
        "index %" PRId64 " is outside weight rows [0, %" PRId64 ")",
        row,
        static_cast<int64_t>(weight.size(0)));
  }

  const SizesType out_sizes[2] = {
      static_cast<SizesType>(num_bags), static_cast<SizesType>(dim)};
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {out_sizes, 2}) == Error::Ok,
      InvalidArgument,
      ret_val);

  // The auxiliary outputs exist for the backward pass, and the ATen CPU kernel
  // only fills the ones each mode needs; match it so shapes agree.
  const SizesType offset2bag_size =
      static_cast<SizesType>(mode == kModeSum ? 0 : num_indices);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(offset2bag, {&offset2bag_size, 1}) == Error::Ok,
      InvalidArgument,
      ret_val);

  const SizesType bag_size_size = static_cast<SizesType>(num_bags);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(bag_size, {&bag_size_size, 1}) == Error::Ok,
      InvalidArgument,
      ret_val);

  if (mode == kModeMax) {
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(max_indices, {out_sizes, 2}) == Error::Ok,
        InvalidArgument,
        ret_val);
  } else {
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(max_indices, {&bag_size_size, 1}) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  if (bag_size.nbytes() > 0) {
    std::memset(bag_size.mutable_data_ptr(), 0, bag_size.nbytes());
  }

  if (mode != kModeSum && num_indices > 0) {
    for (const auto bag : c10::irange(num_bags)) {
      const int64_t start = index_at(offsets, bag);
      const int64_t end =
          bag + 1 < offsets.numel() ? index_at(offsets, bag + 1) : num_indices;
      for (int64_t i = start; i < end; ++i) {
        set_index(offset2bag, i, bag);
      }
    }
  }

  if (out.numel() == 0) {
    return ret_val;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "_embedding_bag.out");

  ET_SWITCH_FLOATHBF16_TYPES(weight.scalar_type(), ctx, name, CTYPE, [&]() {
    const CTYPE* const weight_data = weight.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
    const CTYPE* const psw_data =
        per_sample_weights.has_value() && per_sample_weights.value().numel() > 0
        ? per_sample_weights.value().const_data_ptr<CTYPE>()
        : nullptr;
    if (max_indices.nbytes() > 0) {
      std::memset(max_indices.mutable_data_ptr(), 0, max_indices.nbytes());
    }

    for (const auto bag : c10::irange(num_bags)) {
      const int64_t start = index_at(offsets, bag);
      const int64_t end =
          bag + 1 < offsets.numel() ? index_at(offsets, bag + 1) : num_indices;

      CTYPE* const bag_out = out_data + bag * dim;
      int64_t count = 0;
      for (int64_t i = start; i < end; ++i) {
        if (index_at(indices, i) != padding_idx) {
          ++count;
        }
      }

      using AccType = std::conditional_t<
          std::is_same_v<CTYPE, executorch::aten::Half> ||
              std::is_same_v<CTYPE, executorch::aten::BFloat16>,
          float,
          CTYPE>;
      for (const auto d : c10::irange(dim)) {
        AccType value = 0;
        bool first = true;
        for (int64_t i = start; i < end; ++i) {
          const int64_t row = index_at(indices, i);
          if (row == padding_idx) {
            continue;
          }
          const AccType element =
              static_cast<AccType>(weight_data[row * dim + d]);
          if (mode == kModeMax) {
            if (first || element > value) {
              value = element;
              set_index(max_indices, bag * dim + d, row);
            }
          } else {
            const AccType scale = psw_data == nullptr
                ? static_cast<AccType>(1)
                : static_cast<AccType>(psw_data[i]);
            value += element * scale;
          }
          first = false;
        }
        if (mode == kModeMean && count > 0) {
          value /= static_cast<AccType>(count);
        }
        bag_out[d] = static_cast<CTYPE>(value);
      }

      // The ATen CPU kernel leaves bag_size at zero for the sum mode, and
      // mirrors it into the 1-D max_indices for the mean mode, where that
      // output carries nothing else.
      if (mode != kModeSum) {
        set_index(bag_size, bag, count);
      }
      if (mode == kModeMean) {
        set_index(max_indices, bag, count);
      }
    }
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
