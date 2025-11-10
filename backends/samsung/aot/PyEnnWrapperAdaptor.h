/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <include/graphgen_c.h>
#include <include/graphgen_common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <executorch/backends/samsung/compile_options_def_generated.h>
#include <executorch/backends/samsung/runtime/logging.h>

#include <iostream>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace torch {
namespace executor {
namespace enn {

class PyEnnWrapper {
 public:
  PyEnnWrapper() {}

  void Init(const py::bytes& compile_opts) {
    graphgen_instance_ = graphgen_create();
    option_buf_ = enn_option::GetEnnExecuTorchOptions(
        compile_opts.cast<std::string_view>().data());
  }

  bool IsNodeSupportedByBackend() {
    return false;
  }

  py::array_t<char> Compile(const py::array_t<char>& model_buffer) {
    if (graphgen_instance_ == nullptr) {
      ENN_LOG_ERROR("Please call `Init()` first before compile.");
      return py::array_t<char>();
    }
    auto soc_name = option_buf_->chipset();
    if (graphgen_initialize_context(graphgen_instance_, soc_name) !=
        GraphGenResult::SUCCESS) {
      ENN_LOG_ERROR(
          "Unsupported Soc (%d), please check your chipset version.", soc_name);
      return py::array_t<char>();
    }

    auto m_buf_info = model_buffer.request();
    auto* model_buf_ptr = reinterpret_cast<uint8_t*>(m_buf_info.ptr);
    NNCBuffer* nnc_buffer = nullptr;
    if (graphgen_generate(
            graphgen_instance_, model_buf_ptr, m_buf_info.size, &nnc_buffer) !=
        GraphGenResult::SUCCESS) {
      ENN_LOG_ERROR("Compile model failed.");
      return py::array_t<char>();
    }

    auto result = py::array_t<char>({nnc_buffer->size}, {sizeof(char)});
    auto result_buf = result.request();
    memcpy(result_buf.ptr, nnc_buffer->addr, nnc_buffer->size);

    graphgen_release_buffer(graphgen_instance_, nnc_buffer);

    return result;
  }

  void Destroy() {
    graphgen_release(graphgen_instance_);
    graphgen_instance_ = nullptr;
  }

  ~PyEnnWrapper() {
    Destroy();
  }

 private:
  // pointer to enn software entry
  void* graphgen_instance_ = nullptr;
  // enn compilation option buf
  const enn_option::EnnExecuTorchOptions* option_buf_ = nullptr;
};
} // namespace enn
} // namespace executor
} // namespace torch
