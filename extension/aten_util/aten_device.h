/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/platform/assert.h>

#include <c10/core/Device.h> // @manual=//caffe2/c10:c10

#include <optional>

// These live apart from aten_bridge.h so that a translation unit compiled with
// USE_ATEN_LIB can convert devices. aten_bridge.h is usable only in portable
// mode: it names torch::executor::ScalarType and torch::executor::Tensor,
// aliases that exec_aten.h defines only when it is NOT building against ATen.

namespace executorch {
namespace extension {

/**
 * Maps an ExecuTorch device onto the PyTorch device naming the same location.
 */
inline c10::Device executorch_to_torch_device(
    executorch::runtime::etensor::Device device) {
  switch (device.type()) {
    case executorch::runtime::etensor::DeviceType::CPU:
      return c10::Device(c10::DeviceType::CPU);
    case executorch::runtime::etensor::DeviceType::CUDA:
      return c10::Device(c10::DeviceType::CUDA, device.index());
  }
  // Aborting on an unknown device rather than falling back to CPU: a wrong
  // label made the host read accelerator memory and segfault.
  ET_CHECK_MSG(
      false,
      "Tensor reports device type %d, which this build cannot map to a PyTorch device",
      static_cast<int>(device.type()));
}

/**
 * Maps a PyTorch device onto the ExecuTorch device naming the same location.
 *
 * Returns nothing for a device this runtime has no type for. That is a valid
 * thing for a caller to ask, so it is reported rather than fatal, and the
 * caller adds the context it has before failing.
 */
inline std::optional<executorch::runtime::etensor::Device>
torch_to_executorch_device(c10::Device device) {
  switch (device.type()) {
    case c10::DeviceType::CPU:
      return executorch::runtime::etensor::Device(
          executorch::runtime::etensor::DeviceType::CPU);
    case c10::DeviceType::CUDA:
      return executorch::runtime::etensor::Device(
          executorch::runtime::etensor::DeviceType::CUDA, device.index());
    default:
      return std::nullopt;
  }
}

} // namespace extension
} // namespace executorch
