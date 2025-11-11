/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>
#include <cstdio>
#include <memory>

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h> // @manual=//caffe2:torch_extension
#include <torch/torch.h> // @manual=//caffe2:torch-cpp-cpu

namespace exir {
namespace {

class DataBuffer {
 private:
  void* buffer_ = nullptr;

 public:
  DataBuffer(pybind11::bytes data, int64_t len) {
    // allocate buffer
    buffer_ = malloc(len);
    // convert data to std::string and copy to buffer
    std::memcpy(buffer_, (std::string{data}).data(), len);
  }
  ~DataBuffer() {
    if (buffer_) {
      free(buffer_);
    }
  }
  DataBuffer(const DataBuffer&) = delete;
  DataBuffer& operator=(const DataBuffer&) = delete;

  void* get() {
    return buffer_;
  }
};
} // namespace

PYBIND11_MODULE(bindings, m) {
  pybind11::class_<DataBuffer>(m, "DataBuffer")
      .def(pybind11::init<pybind11::bytes, int64_t>());
  m.def(
      "convert_to_tensor",
      [&](DataBuffer& data_buffer,
          const int64_t scalar_type,
          const std::vector<int64_t>& sizes,
          const std::vector<int64_t>& strides) {
        at::ScalarType type_option = static_cast<at::ScalarType>(scalar_type);
        auto opts = torch::TensorOptions().dtype(type_option);

        // get tensor from memory using metadata
        torch::Tensor result =
            torch::from_blob(data_buffer.get(), sizes, strides, opts);
        return result;
      });
}
} // namespace exir
