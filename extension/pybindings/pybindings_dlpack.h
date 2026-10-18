/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>

#include <pybind11/pybind11.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

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

namespace executorch::extension::pybindings {

namespace py = pybind11;

class TorchDLPackBuffer final {
 public:
  static py::capsule create(
      const executorch::aten::Tensor& tensor,
      py::object owner) {
    auto holder = std::unique_ptr<TorchDLPackBuffer>(
        new TorchDLPackBuffer(tensor, std::move(owner)));
    py::capsule capsule(&holder->managed_, "dltensor", [](PyObject* capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* managed = static_cast<dlpack::ManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        managed->deleter(managed);
      }
    });
    holder.release();
    return capsule;
  }

 private:
  TorchDLPackBuffer(const executorch::aten::Tensor& tensor, py::object owner)
      : owner_(std::move(owner)), nbytes_(tensor.nbytes()) {
    managed_.dl_tensor = {
        const_cast<void*>(tensor.const_data_ptr()),
        {tensor.device().is_cpu() ? dlpack::CPU : dlpack::CUDA,
         tensor.device().index()},
        1,
        {dlpack::UInt, 8, 1},
        &nbytes_,
        &stride_,
        0};
    managed_.manager_ctx = this;
    managed_.deleter = [](dlpack::ManagedTensor* managed) {
      py::gil_scoped_acquire gil;
      delete static_cast<TorchDLPackBuffer*>(managed->manager_ctx);
    };
  }

  py::object owner_;
  int64_t nbytes_;
  int64_t stride_ = 1;
  dlpack::ManagedTensor managed_{};
};

} // namespace executorch::extension::pybindings
