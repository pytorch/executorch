/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "cortex_m_ops_common.h"
extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

// During AOT phase, quantized_linear_fusion_pass allocates total buffer
// and passes in as 'Tensor'. (Total buffer = 8-byte header + x bytes)
// ┌─────────────────┬─────────────────────────────────────┐
// │ KernelSum Header│        CMSIS Workspace              │
// │    (8 bytes)    │         (x bytes)                   │
// └─────────────────┴─────────────────────────────────────┘
//          │                           │
//          │                           └─> Passed to CMSIS API
//          │
//          └─> State for kernel sum

// C++ Runtime:
// ┌─────────────────┬─────────────────────────────────────┐
// │ KernelSum Header│        CMSIS Workspace              │
// │    (8 bytes)    │         (x bytes)                   │
// └─────────────────┴─────────────────────────────────────┘
// ^                 ^
// │                 │
// scratch_ptr       cmsis_workspace_ptr
// │                 │
// ▼                 ▼
//             arm_vector_sum_s8() writes kernel sums (with bias if avail):
//             [sum₀+bias₀][sum₁+bias₁][sum₂+bias₂]...[sum_{n-1}+bias_{n-1}]
//             (n * 4-byte int32_t values = x bytes)
//
// - n = out_features (number of output features)
// - x = n * 4 bytes (total CMSIS buffer size)
// - Total buffer = 8 + x bytes

class CMSISScratchBufferContext final {
 public:
  CMSISScratchBufferContext(
      Tensor& scratch_buffer,
      const Tensor& weights,
      const Tensor& weight_zero_point,
      const torch::executor::optional<Tensor>& bias)
      : scratch_ptr_(scratch_buffer.mutable_data_ptr<int8_t>()),
        total_size_(scratch_buffer.size(0)),
        base_ptr_(reinterpret_cast<uint8_t*>(scratch_ptr_)),
        in_features_(weights.size(1)),
        out_features_(weights.size(0)),
        is_per_channel_(weight_zero_point.numel() > 1),
        weight_data_offset_(calculate_offset(weights.const_data_ptr<int8_t>())),
        weight_zp_data_offset_(
            calculate_offset(weight_zero_point.const_data_ptr<int32_t>())),
        bias_data_offset_(
            bias.has_value()
                ? calculate_offset(bias.value().const_data_ptr<int32_t>())
                : 0),
        header_(reinterpret_cast<KernelSumHeader*>(scratch_ptr_)),
        cmsis_workspace_ptr_(scratch_ptr_ + KERNEL_SUM_HEADER_SIZE) {
    cmsis_nn_dims filter_dims = {in_features_, 1, 1, out_features_};
    validate_size(filter_dims);
  }

  cmsis_nn_context get_cmsis_ctx() const {
    cmsis_nn_context ctx;
    ET_CHECK_MSG(
        reinterpret_cast<uintptr_t>(cmsis_workspace_ptr_) % 4 == 0,
        "CMSIS workspace not 4-byte aligned");
    ctx.buf = cmsis_workspace_ptr_;
    ctx.size = get_cmsis_workspace_size();
    return ctx;
  }

  bool is_kernel_sum_updated() const {
    return header_->updated;
  }

  void compute_kernel_sums_if_needed() {
    if (!header_->updated) {
      arm_vector_sum_s8(
          reinterpret_cast<int32_t*>(cmsis_workspace_ptr_),
          in_features_,
          out_features_,
          get_weight_data(),
          get_weight_zp_data()[0],
          0,
          get_bias_data());
      header_->updated = true;
      ET_LOG(
          Info,
          "Computed kernel sums. [required_bytes : %d]",
          header_->required_size);
    }
  }

  const int8_t* get_weight_data() const {
    return reinterpret_cast<const int8_t*>(base_ptr_ + weight_data_offset_);
  }

  const int32_t* get_weight_zp_data() const {
    return reinterpret_cast<const int32_t*>(base_ptr_ + weight_zp_data_offset_);
  }

  const int32_t* get_bias_data() const {
    return bias_data_offset_ == 0
        ? nullptr
        : reinterpret_cast<const int32_t*>(base_ptr_ + bias_data_offset_);
  }

  bool is_per_channel_quant() const {
    return is_per_channel_;
  }
  int32_t get_in_features() const {
    return in_features_;
  }
  int32_t get_out_features() const {
    return out_features_;
  }

 private:
  static constexpr size_t KERNEL_SUM_HEADER_SIZE = 8;

  // Header for kernel sum computation state only
  struct KernelSumHeader {
    bool updated = false;
    int32_t required_size = 0;
  };
  static_assert(
      sizeof(KernelSumHeader) == KERNEL_SUM_HEADER_SIZE,
      "KernelSumHeader must be exactly 8 bytes");

  int8_t* scratch_ptr_;
  size_t total_size_;
  uint8_t* base_ptr_;

  // Context members
  const int32_t in_features_;
  const int32_t out_features_;
  const bool is_per_channel_;
  const uint32_t weight_data_offset_;
  const uint32_t weight_zp_data_offset_;
  const uint32_t bias_data_offset_;

  KernelSumHeader* header_;
  int8_t* cmsis_workspace_ptr_;

  uint32_t calculate_offset(const void* ptr) const {
    if (ptr == nullptr)
      return 0;

    const uint8_t* ptr_bytes = reinterpret_cast<const uint8_t*>(ptr);
    ET_CHECK_MSG(ptr_bytes >= base_ptr_, "Pointer is before base address");

    const std::ptrdiff_t offset = ptr_bytes - base_ptr_;
    ET_CHECK_MSG(
        offset >= 0 && offset <= UINT32_MAX, "Offset out of valid range");
    return static_cast<uint32_t>(offset);
  }

  size_t get_cmsis_workspace_size() const {
    return total_size_ - KERNEL_SUM_HEADER_SIZE;
  }

  void validate_size(const cmsis_nn_dims& filter_dims) const {
    header_->required_size =
        arm_fully_connected_s8_get_buffer_size(&filter_dims);

    ET_CHECK_MSG(
        get_cmsis_workspace_size() >=
            static_cast<size_t>(header_->required_size),
        "Scratch buffer size %zu insufficient for required size %d",
        get_cmsis_workspace_size(),
        header_->required_size);
  }
};

} // namespace native
} // namespace cortex_m
