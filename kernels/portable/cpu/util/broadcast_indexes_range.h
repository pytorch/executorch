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

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>

namespace torch::executor {

namespace internal {
template <std::size_t kNumInputs>
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
      : output_dim_(output.dim()),
        output_shape_(output.sizes()),
        effective_input_broadcast_strides_{
            effective_input_broadcast_stride(output, args)...} {
    static_assert(
        sizeof...(args) == kNumInputs && (std::is_same_v<Args, Tensor> && ...),
        "BroadcastIndexesIterator constructor requires kNumInputs input tensor"
        "arguments!");
  }

  struct make_end_t {
    explicit constexpr make_end_t() = default;
  };

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
    // TODO: add optimization for particular input tensors not being
    // broadcasted?
    for (auto ii = output_dim_ - 1; ii >= 0; --ii) {
      // You might wonder what happens if output_shape_[ii] == 0. In that case,
      // output.numel() would be 0, and thus the iterator would be the end()
      // iterator, which is not legal to increment.
      if ET_UNLIKELY (delinearized_output_index_[ii] == output_shape_[ii] - 1) {
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

  difference_type operator-(const BroadcastIndexesIterator& rhs) const {
    return difference_type(output_index() - rhs.output_index());
  }

 private:
  ssize_t output_index() const {
    return current_indexes_[0];
  }

  ssize_t& output_index() {
    return current_indexes_[0];
  }

  std::array<exec_aten::SizesType, executorch::runtime::kTensorDimensionLimit>
  effective_input_broadcast_stride(const Tensor& output, const Tensor& t)
      const {
    std::array<exec_aten::SizesType, executorch::runtime::kTensorDimensionLimit>
        result = {0};
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
    return result;
  }

  // The 0th entry is the current linear index into the output,
  // followed by kNumInputs input indexes.
  std::array<ssize_t, kNumInputs + 1> current_indexes_ = {0};
  using ShapeType = std::
      array<exec_aten::SizesType, executorch::runtime::kTensorDimensionLimit>;
  ShapeType delinearized_output_index_ = {0};
  ssize_t output_dim_;
  ArrayRef<exec_aten::SizesType> output_shape_;
  // The linear index for a broadcast tensor is
  // sum(delinearized_output_index_[i] * input_stride_[i] if
  // padded_input_shape_[i] != 1 else 0), where padded_input_shape is
  // input.sizes() with leading 1s added to make its size equal to
  // output_dim. This is straightforwardly implementable with an
  // adjusted stride array that contains 0s where the padded input
  // shape would contain 1s.
  std::array<ShapeType, kNumInputs> effective_input_broadcast_strides_ = {
      {{0}}};
};
} // namespace internal

// Efficient mechanism for looping over the index space for an output
// tensor and kNumInputs possibly-broadcasted input tensors. Use as follows:
//
// auto* output_data = output.mutable_data_ptr<OutputType>();
// const auto* a_data = a.mutable_data_ptr<AType>();
// const auto* b_data = b.mutable_data_ptr<BType>();
// for (const auto [output_index, a_index, b_index] :
// BroadcastIndexesRange<2>(output, a, b)) {
//   // Access output_data[output_index], a_data[a_index], and b_data[b_index].
// }
//
// (where OutputType, AType, and BType are known concrete types.)
//
// Unlike looping using delinearize_index() and
// linearize_access_indexes(), BroadcastIndexesRange avoids expensive
// division and modulo operations on each iteration.
template <std::size_t kNumInputs>
class BroadcastIndexesRange {
 public:
  using iterator = internal::BroadcastIndexesIterator<kNumInputs>;

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
} // namespace torch::executor
