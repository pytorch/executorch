/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <utility>

#include <executorch/kernels/portable/cpu/util/delinearize_index.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>

namespace torch::executor {

namespace internal {
// NOTE: we bake ArrayRef iterators being pointers into the return
// type here because we assume that iterators are portable across
// ArrayRef copies.
inline const Tensor::SizesType* arrayref_begin_ignoring_leading_1s(
    ArrayRef<Tensor::SizesType> arr) {
  return std::find_if(
      arr.begin(), arr.end(), [](Tensor::SizesType x) { return x != 1; });
}

inline bool sizes_match_ignoring_leading_1s(
    ArrayRef<Tensor::SizesType> lhs,
    ArrayRef<Tensor::SizesType> rhs) {
  auto lhs_begin = arrayref_begin_ignoring_leading_1s(lhs);
  auto lhs_end = lhs.end();

  auto rhs_begin = arrayref_begin_ignoring_leading_1s(rhs);
  auto rhs_end = rhs.end();

  return ((lhs_end - lhs_begin) == (rhs_end - rhs_begin)) &&
      std::equal(lhs_begin, lhs_end, rhs_begin);
}

// Whether the same memory offset addresses the same logical element in
// both tensors, which requires matching sizes and matching strides for
// every dim whose size is not 1. Not inlined so that
// BroadcastIndexesIterator's constructor stays small enough to inline,
// which lets callers keep the iterator's indexes in registers.
ET_NOINLINE inline bool layouts_match_ignoring_leading_1s(
    ArrayRef<Tensor::SizesType> lhs_sizes,
    ArrayRef<Tensor::StridesType> lhs_strides,
    ArrayRef<Tensor::SizesType> rhs_sizes,
    ArrayRef<Tensor::StridesType> rhs_strides) {
  if (!sizes_match_ignoring_leading_1s(lhs_sizes, rhs_sizes)) {
    return false;
  }
  const auto lhs_dim = lhs_sizes.size();
  const auto rhs_dim = rhs_sizes.size();
  for (const auto ii : c10::irange(1, std::min(lhs_dim, rhs_dim) + 1)) {
    if (lhs_sizes[lhs_dim - ii] != 1 &&
        lhs_strides[lhs_dim - ii] != rhs_strides[rhs_dim - ii]) {
      return false;
    }
  }
  return true;
}

template <
    std::size_t kNumInputs,
    bool support_noncontiguous_input_tensors = false>
class BroadcastIndexesIterator {
 public:
  using difference_type = ssize_t;
  using value_type = std::array<ssize_t, kNumInputs + 1>;
  using reference = const value_type&;
  using pointer = const value_type*;
  using iterator_category = std::forward_iterator_tag;

  BroadcastIndexesIterator() = default;

  template <typename... Args>
  explicit BroadcastIndexesIterator(const Tensor& output, const Args&... args)
      : output_dim_or_zero_if_no_broadcasting_(
            !support_noncontiguous_input_tensors &&
                    (layouts_match_ignoring_leading_1s(
                         args.sizes(),
                         args.strides(),
                         output.sizes(),
                         output.strides()) &&
                     ...)
                ? 0
                : output.dim()) {
    static_assert(
        sizeof...(args) == kNumInputs && (std::is_same_v<Args, Tensor> && ...),
        "BroadcastIndexesIterator constructor requires kNumInputs input tensor"
        "arguments!");
    if (support_noncontiguous_input_tensors ||
        output_dim_or_zero_if_no_broadcasting_ != 0) {
      const auto output_dims = output_dims_in_memory_order(output.strides());
      for (const auto ii : c10::irange(output.dim())) {
        output_shape_[ii] = output.size(output_dims[ii]);
      }
      effective_input_broadcast_strides_ = {
          effective_input_broadcast_stride(output, args, output_dims)...};
    }
  }

  template <typename... Args>
  explicit BroadcastIndexesIterator(
      ArrayRef<Tensor::SizesType> output_sizes,
      const Args&... args) {
    static_assert(
        sizeof...(args) == kNumInputs && (std::is_same_v<Args, Tensor> && ...),
        "BroadcastIndexesIterator constructor requires kNumInputs input tensor"
        "arguments!");
    std::array<Tensor::StridesType, executorch::runtime::kTensorDimensionLimit>
        output_strides{};
    Tensor::StridesType stride = 1;
    for (auto ii = static_cast<ssize_t>(output_sizes.size()) - 1; ii >= 0;
         --ii) {
      output_strides[ii] = stride;
      stride *= output_sizes[ii];
    }
    output_dim_or_zero_if_no_broadcasting_ =
        !support_noncontiguous_input_tensors &&
            (layouts_match_ignoring_leading_1s(
                 args.sizes(),
                 args.strides(),
                 output_sizes,
                 {output_strides.data(), output_sizes.size()}) &&
             ...)
        ? 0
        : output_sizes.size();
    if (support_noncontiguous_input_tensors ||
        output_dim_or_zero_if_no_broadcasting_ != 0) {
      effective_input_broadcast_strides_ = {
          effective_input_broadcast_stride(output_sizes.size(), args)...};
      std::copy(
          output_sizes.begin(), output_sizes.end(), output_shape_.begin());
    }
  }

  struct make_end_t {
    explicit constexpr make_end_t() = default;
  };

  template <typename... Args>
  BroadcastIndexesIterator(
      make_end_t,
      ssize_t output_numel,
      const Args&... args)
      : current_indexes_{
            output_numel,
            0,
        } {}

  template <typename... Args>
  BroadcastIndexesIterator(make_end_t, const Tensor& t, const Args&... args)
      : current_indexes_{
            t.numel(),
            0,
        } {}

  bool operator==(const BroadcastIndexesIterator& rhs) const {
    return output_index() == rhs.output_index();
  }

  bool operator!=(const BroadcastIndexesIterator& rhs) const {
    return !operator==(rhs);
  }

  reference operator*() const {
    return current_indexes_;
  }

  pointer operator->() const {
    return &current_indexes_;
  }

  BroadcastIndexesIterator& operator++() {
    output_index()++;
    if (output_dim_or_zero_if_no_broadcasting_ == 0) {
      std::fill(
          current_indexes_.begin() + 1, current_indexes_.end(), output_index());
      return *this;
    }
    // TODO: add optimization for particular input tensors not being
    // broadcasted?
    for (auto ii = output_dim_or_zero_if_no_broadcasting_ - 1; ii >= 0; --ii) {
      // You might wonder what happens if output_shape_[ii] == 0. In
      // that case, output.numel() would be 0, and thus we would have
      // begin() == end() and no iteration.
      if ET_UNLIKELY (
          static_cast<exec_aten::SizesType>(delinearized_output_index_[ii]) ==
          output_shape_[ii] - 1) {
        const auto old_delinearized_output_index_item =
            delinearized_output_index_[ii];
        delinearized_output_index_[ii] = 0;
        for (const auto jj : c10::irange(1, kNumInputs + 1)) {
          current_indexes_[jj] -= old_delinearized_output_index_item *
              effective_input_broadcast_strides_[jj - 1][ii];
        }
      } else {
        delinearized_output_index_[ii]++;
        for (const auto jj : c10::irange(1, kNumInputs + 1)) {
          current_indexes_.at(jj) +=
              effective_input_broadcast_strides_[jj - 1][ii];
        }
        break;
      }
    }
    return *this;
  }

  BroadcastIndexesIterator operator++(int) {
    auto it = *this;
    operator++();
    return it;
  }

  BroadcastIndexesIterator& operator+=(difference_type n) {
    if (n <= 3) {
      std::advance(*this, n);
      return *this;
    }

    output_index() += n;
    if (output_dim_or_zero_if_no_broadcasting_ == 0) {
      std::fill(
          current_indexes_.begin() + 1, current_indexes_.end(), output_index());
      return *this;
    }
    delinearize_index(
        output_index(),
        {output_shape_.data(),
         static_cast<size_t>(output_dim_or_zero_if_no_broadcasting_)},
        delinearized_output_index_.data(),
        delinearized_output_index_.size());
    for (const auto ii : c10::irange(1, kNumInputs + 1)) {
      current_indexes_[ii] = 0;
      for (const auto jj :
           c10::irange(output_dim_or_zero_if_no_broadcasting_)) {
        current_indexes_[ii] += delinearized_output_index_[jj] *
            effective_input_broadcast_strides_[ii - 1][jj];
      }
    }
    return *this;
  }

  BroadcastIndexesIterator operator+(difference_type n) {
    auto it = *this;
    it += n;
    return it;
  }

  difference_type operator-(const BroadcastIndexesIterator& rhs) const {
    return difference_type(output_index() - rhs.output_index());
  }

 private:
  using ShapeType =
      std::array<std::size_t, executorch::runtime::kTensorDimensionLimit>;

  ssize_t output_index() const {
    return current_indexes_[0];
  }

  ssize_t& output_index() {
    return current_indexes_[0];
  }

  // The output's dims from outermost to innermost in memory. Iterating over
  // the dims in this order visits the output's elements in memory order.
  static ShapeType output_dims_in_memory_order(
      ArrayRef<Tensor::StridesType> output_strides) {
    ShapeType dims = {0};
    for (const auto ii : c10::irange(output_strides.size())) {
      dims[ii] = ii;
      for (auto jj = ii;
           jj > 0 && output_strides[dims[jj - 1]] < output_strides[dims[jj]];
           --jj) {
        std::swap(dims[jj - 1], dims[jj]);
      }
    }
    return dims;
  }

  ShapeType effective_input_broadcast_stride(
      const Tensor& output,
      const Tensor& t,
      const ShapeType& output_dims) const {
    ShapeType result = {0};
    ET_CHECK_MSG(
        t.dim() <= output.dim(),
        "input to broadcasting op should have dim at most output dim, but %d > %d!",
        (int)t.dim(),
        (int)output.dim());

    const auto num_leading_ones = output.dim() - t.dim();
    for (const auto idx : c10::irange(num_leading_ones)) {
      result[idx] = 0;
    }
    const auto t_sizes = t.sizes();
    const auto t_strides = t.strides();
    for (const auto idx :
         c10::irange(num_leading_ones, num_leading_ones + t.dim())) {
      result[idx] = t_sizes[idx - num_leading_ones] == 1
          ? 0
          : t_strides[idx - num_leading_ones];
    }
    ShapeType result_in_output_dims_order = {0};
    for (const auto idx : c10::irange(output.dim())) {
      result_in_output_dims_order[idx] = result[output_dims[idx]];
    }
    return result_in_output_dims_order;
  }

  // Same as the overload above, for a contiguous output given by its sizes,
  // whose dims need no reordering. The body is repeated rather than shared so
  // the tensor overload generates the same code whether or not this one is
  // used.
  ShapeType effective_input_broadcast_stride(
      ssize_t output_dim,
      const Tensor& t) const {
    ShapeType result = {0};
    ET_CHECK_MSG(
        t.dim() <= output_dim,
        "input to broadcasting op should have dim at most output dim, but %d > %d!",
        (int)t.dim(),
        (int)output_dim);

    const auto num_leading_ones = output_dim - t.dim();
    for (const auto idx : c10::irange(num_leading_ones)) {
      result[idx] = 0;
    }
    const auto t_sizes = t.sizes();
    const auto t_strides = t.strides();
    for (const auto idx :
         c10::irange(num_leading_ones, num_leading_ones + t.dim())) {
      result[idx] = t_sizes[idx - num_leading_ones] == 1
          ? 0
          : t_strides[idx - num_leading_ones];
    }
    return result;
  }

  // The 0th entry is the current linear index into the output,
  // followed by kNumInputs input indexes.
  std::array<ssize_t, kNumInputs + 1> current_indexes_ = {0};
  ShapeType delinearized_output_index_ = {0};
  ssize_t output_dim_or_zero_if_no_broadcasting_;
  std::array<Tensor::SizesType, executorch::runtime::kTensorDimensionLimit>
      output_shape_;
  // The linear index for a broadcast tensor is
  // sum(delinearized_output_index_[i] * input_stride_[i] if
  // padded_input_shape_[i] != 1 else 0), where padded_input_shape is
  // input.sizes() with leading 1s added to make its size equal to
  // output_dim. This is straightforwardly implementable with an
  // adjusted stride array that contains 0s where the padded input
  // shape would contain 1s.
  std::array<ShapeType, kNumInputs> effective_input_broadcast_strides_;
};

// When there is only 1 input and no noncontiguous tensor support
// required, there is no actual broadcasting to do.
template <>
class BroadcastIndexesIterator<1, false> {
 public:
  using difference_type = ssize_t;
  using value_type = std::array<ssize_t, 2>;
  using reference = value_type;
  using pointer = const value_type*;
  using iterator_category = std::forward_iterator_tag;

  BroadcastIndexesIterator() = default;

  explicit BroadcastIndexesIterator(
      [[maybe_unused]] const Tensor& output,
      [[maybe_unused]] const Tensor& input) {}

  struct make_end_t {
    explicit constexpr make_end_t() = default;
  };

  BroadcastIndexesIterator(
      make_end_t,
      const Tensor& output,
      [[maybe_unused]] const Tensor& input)
      : current_indexes_({output.numel(), output.numel()}) {}

  explicit BroadcastIndexesIterator(
      [[maybe_unused]] ArrayRef<Tensor::SizesType> output_sizes,
      [[maybe_unused]] const Tensor& input) {}

  BroadcastIndexesIterator(
      make_end_t,
      ssize_t output_numel,
      [[maybe_unused]] const Tensor& input)
      : current_indexes_({output_numel, output_numel}) {}

  bool operator==(const BroadcastIndexesIterator& rhs) const {
    return current_index() == rhs.current_index();
  }

  bool operator!=(const BroadcastIndexesIterator& rhs) const {
    return current_index() != rhs.current_index();
  }

  reference operator*() const {
    return current_indexes_;
  }

  pointer operator->() const {
    return &current_indexes_;
  }

  BroadcastIndexesIterator& operator++() {
    add_to_current_index(1);
    return *this;
  }

  BroadcastIndexesIterator operator++(int) {
    auto it = *this;
    operator++();
    return it;
  }

  BroadcastIndexesIterator& operator+=(difference_type n) {
    add_to_current_index(n);
    return *this;
  }

  BroadcastIndexesIterator operator+(difference_type n) {
    auto it = *this;
    it += n;
    return it;
  }

  difference_type operator-(const BroadcastIndexesIterator& rhs) const {
    return difference_type(current_index() - rhs.current_index());
  }

 private:
  ssize_t current_index() const {
    return current_indexes_[0];
  }

  void add_to_current_index(ssize_t n) {
    current_indexes_[0] += n;
    current_indexes_[1] = current_indexes_[0];
  }
  value_type current_indexes_ = {{0, 0}};
};
} // namespace internal

/**
 * Efficient mechanism for looping over the index space for an output
 * tensor and kNumInputs possibly-broadcasted input tensors. Use as follows:
 *
 * auto* output_data = output.mutable_data_ptr<OutputType>();
 * const auto* a_data = a.mutable_data_ptr<AType>();
 * const auto* b_data = b.mutable_data_ptr<BType>();
 * for (const auto [output_index, a_index, b_index] :
 *      BroadcastIndexesRange<2>(output, a, b)) {
 *   // Access output_data[output_index], a_data[a_index], and b_data[b_index].
 * }
 *
 * (where OutputType, AType, and BType are known concrete types.)
 *
 * Unlike looping using delinearize_index() and
 * linearize_access_indexes(), BroadcastIndexesRange avoids expensive
 * division and modulo operations on each iteration.
 *
 * The iterators visit the output's elements in memory order, so the output
 * may have any dense layout (e.g. channels last) and its index is the number
 * of elements visited so far. Input indexes are memory offsets that respect
 * strides.
 *
 * By default, the iterators skip index computation when every input has the
 * output's sizes and strides. The support_noncontiguous_input_tensors
 * argument disables this optimization. With one input and
 * support_noncontiguous_input_tensors=false, the input is assumed to have
 * the output's shape and strides and is not broadcast.
 */
template <
    std::size_t kNumInputs,
    bool support_noncontiguous_input_tensors = false>
class BroadcastIndexesRange {
 public:
  using iterator = internal::
      BroadcastIndexesIterator<kNumInputs, support_noncontiguous_input_tensors>;

  template <typename... Args>
  BroadcastIndexesRange(const Tensor& output, const Args&... args)
      : tensors_{&output, (&args)...} {}

  iterator begin() const {
    return std::apply(
        [](const auto&... args) { return iterator((*args)...); }, tensors_);
  }

  iterator end() const {
    return std::apply(
        [](const auto&... args) {
          return iterator(typename iterator::make_end_t(), (*args)...);
        },
        tensors_);
  }

 private:
  std::array<const Tensor*, kNumInputs + 1> tensors_;
};

/**
 * Like BroadcastIndexesRange, but takes the output's sizes instead of an
 * output tensor and treats the output as contiguous. The sizes' storage must
 * stay alive and unchanged while the range or its iterators are in use. With
 * one input and support_noncontiguous_input_tensors=false, the input is
 * assumed to be contiguous with the output's shape and is not broadcast.
 */
template <
    std::size_t kNumInputs,
    bool support_noncontiguous_input_tensors = false>
class BroadcastIndexesShapeRange {
 public:
  using iterator = internal::
      BroadcastIndexesIterator<kNumInputs, support_noncontiguous_input_tensors>;

  template <typename... Args>
  BroadcastIndexesShapeRange(
      ArrayRef<Tensor::SizesType> output_sizes,
      const Args&... args)
      : output_sizes_(output_sizes),
        output_numel_(compute_numel(output_sizes.data(), output_sizes.size())),
        inputs_{(&args)...} {}

  iterator begin() const {
    return begin_impl(std::make_index_sequence<kNumInputs>());
  }

  iterator end() const {
    return end_impl(std::make_index_sequence<kNumInputs>());
  }

 private:
  template <std::size_t... I>
  iterator begin_impl(std::index_sequence<I...>) const {
    return iterator(output_sizes_, (*inputs_[I])...);
  }

  template <std::size_t... I>
  iterator end_impl(std::index_sequence<I...>) const {
    return iterator(
        typename iterator::make_end_t(), output_numel_, (*inputs_[I])...);
  }

  ArrayRef<Tensor::SizesType> output_sizes_;
  ssize_t output_numel_;
  std::array<const Tensor*, kNumInputs> inputs_;
};
} // namespace torch::executor
