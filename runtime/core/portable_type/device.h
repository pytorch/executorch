/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {
namespace etensor {

/// Denotes the specific genre of compute device.
/// Subset of https://github.com/pytorch/pytorch/blob/main/c10/core/Device.h
enum class DeviceType : int8_t {
  CPU = 0,
};

/// An index representing a specific device; For cpu it should always be -1 or 0
using DeviceIndex = int8_t;

/**
 * An abstraction for the compute device on which a tensor is located.
 * ExecuTorch doesn't allow dynamic dispatching based on device, so this type is
 * just a skeleton to allow certain kernels that expect device as an
 * argument to still be run.
 *
 * In ExecuTorch this is always expected to be CPU.
 */
struct Device final {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {}

  /// Returns the type of device this is. Only CPU is supported.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  /// Returns the device index. Always 0 if specified or -1 if not provided.
  DeviceIndex index() const noexcept {
    ET_CHECK(index_ == 0 || index_ == -1);
    return index_;
  }

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Device;
using ::executorch::runtime::etensor::DeviceType;
} // namespace executor
} // namespace torch
