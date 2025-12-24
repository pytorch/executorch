/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim::c10 {

/// An index representing a specific device; e.g., the 1 in GPU 1.
/// A DeviceIndex is not independently meaningful without knowing
/// the DeviceType it is associated; try to use Device rather than
/// DeviceIndex directly.
using DeviceIndex = int8_t;

/// Represents a compute device on which a tensor is located.
/// A device is uniquely identified by a type (e.g., CPU) and a device index.
struct Device final {
  using Type = DeviceType;

  /// Constructs a new Device from a DeviceType and an optional device index.
  /// @param type The type of device.
  /// @param index The device index. For CPU, this should be -1 or 0.
  /* implicit */
  Device(DeviceType type, DeviceIndex index = -1) : type_(type), index_(index) {
    validate();
  }

  /// Constructs a Device from a string description.
  /// The string must be "cpu" or "cpu:0".
  /* implicit */ Device(const std::string& device_string) : Device(Type::CPU) {
    ET_CHECK_MSG(!device_string.empty(), "Device string must not be empty");

    if (device_string == "cpu" || device_string == "CPU") {
      type_ = DeviceType::CPU;
      index_ = -1;
    } else if (
        device_string == "cpu:0" || device_string == "CPU:0" ||
        device_string == "cpu:1" || device_string == "CPU:1") {
      type_ = DeviceType::CPU;
      index_ = static_cast<DeviceIndex>(device_string.back() - '0');
    } else {
      ET_CHECK_MSG(
          false,
          "Invalid device string: %s. Currently only 'cpu' is supported.",
          device_string.c_str());
    }
    validate();
  }

  /// Returns true if the type and index of this Device matches that of other.
  bool operator==(const Device& other) const noexcept {
    return this->type_ == other.type_ && this->index_ == other.index_;
  }

  /// Returns true if the type or index of this Device differs from that of
  /// other.
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  /// Sets the device index.
  void set_index(DeviceIndex index) {
    index_ = index;
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns the device index.
  DeviceIndex index() const noexcept {
    return index_;
  }

  /// Returns true if the device has a non-default index.
  bool has_index() const noexcept {
    return index_ != -1;
  }

  /// Returns true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  /// Returns a string representation of the device (e.g., "cpu" or "cpu:0").
  std::string str() const {
    std::string str = DeviceTypeName(type(), /* lower_case */ true);
    if (has_index()) {
      str.push_back(':');
      str.append(std::to_string(index()));
    }
    return str;
  }

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;

  void validate() {
    ET_DCHECK_MSG(
        index_ >= -1,
        "Device index must be -1 or non-negative, got %d",
        static_cast<int>(index_));
    ET_DCHECK_MSG(
        !is_cpu() || index_ <= 0,
        "CPU device index must be -1 or zero, got %d",
        static_cast<int>(index_));
  }
};

inline std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}

} // namespace executorch::backends::aoti::slim::c10

namespace std {
template <>
struct hash<executorch::backends::aoti::slim::c10::Device> {
  size_t operator()(
      executorch::backends::aoti::slim::c10::Device d) const noexcept {
    static_assert(
        sizeof(executorch::backends::aoti::slim::c10::DeviceType) == 1,
        "DeviceType is not 8-bit");
    static_assert(
        sizeof(executorch::backends::aoti::slim::c10::DeviceIndex) == 1,
        "DeviceIndex is not 8-bit");
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()))
            << 16 |
        static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
} // namespace std
