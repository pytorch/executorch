// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdint>
#include <cstdio>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace exir {
namespace {
pybind11::bytes copy_buffer(intptr_t data_ptr, int num_bytes) {
  std::string str(reinterpret_cast<const char*>(data_ptr), num_bytes);
  return pybind11::bytes(str);
}

// buffers x and y are both num_bytes long
pybind11::bool_
equal_buffers(intptr_t data_ptr_x, intptr_t data_ptr_y, int num_bytes) {
  return pybind11::bool_(
      std::memcmp(
          reinterpret_cast<const char*>(data_ptr_x),
          reinterpret_cast<const char*>(data_ptr_y),
          num_bytes) == 0);
}

} // namespace

PYBIND11_MODULE(bindings, m) {
  m.def("copy_buffer", &copy_buffer);
  m.def("equal_buffers", &equal_buffers);
}

} // namespace exir
