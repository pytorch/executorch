/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::extension::pybindings::dlpack {

// Stable ABI types from the DLPack specification. Keeping the small ABI here
// avoids adding a build-time dependency to the Python bindings.
enum DeviceType : int32_t {
  CPU = 1,
  CUDA = 2,
};

enum DataTypeCode : uint8_t {
  Int = 0,
  UInt = 1,
  Float = 2,
  Complex = 5,
  BFloat = 4,
  Bool = 6,
};

struct Device {
  DeviceType device_type;
  int32_t device_id;
};

struct DataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct Tensor {
  void* data;
  Device device;
  int32_t ndim;
  DataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct ManagedTensor {
  Tensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(ManagedTensor* self);
};

} // namespace executorch::extension::pybindings::dlpack
