/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace runtime {
namespace etensor {

/// Represents the type of compute device.
/// Note: ExecuTorch Device is distinct from PyTorch Device.
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
};

/// Total number of device types, used for fixed-size registry arrays.
constexpr size_t kNumDeviceTypes = 2;

/// An index representing a specific device; e.g. GPU 0 vs GPU 1.
/// -1 means the default/unspecified device for that type.
using DeviceIndex = int8_t;

/**
 * An abstraction for the compute device on which a tensor is located.
 *
 * Tensors carry a Device to express where their underlying data resides
 * (e.g. CPU host memory vs CUDA device memory). The runtime uses this to
 * dispatch memory allocation to the appropriate device allocator.
 */
struct Device final {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {}

  /// Returns the type of device the tensor data resides on.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  /// Returns the device index, or -1 if default/unspecified.
  DeviceIndex index() const noexcept {
    return index_;
  }

  bool operator==(const Device& other) const noexcept {
    return type_ == other.type_ && index_ == other.index_;
  }

  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
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
