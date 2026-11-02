/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/executor/method.h>

namespace executorch {
namespace examples {
namespace espressif {

struct EtModelInfo {
  int32_t input_width;
  int32_t input_height;
  size_t classes;
  size_t planned_bytes;
  uint32_t delegates;
  size_t delegate_arena_bytes;
};

class EtModel {
 public:
  bool load(const uint8_t* program, size_t size);
  const EtModelInfo& info() const {
    return info_;
  }
  bool set_input_rgb888(
      const uint8_t* rgb,
      const float mean[3],
      const float stddev[3]);
  bool run();
  int64_t last_run_us() const {
    return last_run_us_;
  }
  int64_t last_delegate_us() const {
    return last_delegate_us_;
  }
  const float* logits() const;

 private:
  using TensorImpl = ::executorch::runtime::etensor::TensorImpl;

  ::executorch::runtime::Method* method_ = nullptr;
  float* input_ = nullptr;
  TensorImpl* input_impl_ = nullptr;
  TensorImpl::SizesType sizes_[4] = {};
  TensorImpl::DimOrderType dim_order_[4] = {};
  TensorImpl::StridesType strides_[4] = {};
  EtModelInfo info_{};
  int64_t last_run_us_ = 0;
  int64_t last_delegate_us_ = 0;
};

} // namespace espressif
} // namespace examples
} // namespace executorch
