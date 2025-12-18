/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnFunctionInterface.h>

#include <dlfcn.h>
#include <string>
namespace executorch {
namespace backends {
namespace qnn {

template <typename Fn>
Fn loadQnnFunction(void* handle, const char* function_name) {
  return reinterpret_cast<Fn>(dlsym(handle, function_name)); // NOLINT
}

class QnnImplementation {
 public:
  using BackendIdType = decltype(QnnInterface_t{}.backendId);

  explicit QnnImplementation(std::string lib_path)
      : lib_path_(std::move(lib_path)){};
  QnnImplementation(const QnnImplementation&) =
      delete; // Delete copy constructor
  QnnImplementation& operator=(const QnnImplementation&) =
      delete; // Delete assignment operator
  ~QnnImplementation();

  executorch::runtime::Error Load(const QnnSaver_Config_t** saver_config);

  const QnnInterface& GetQnnInterface() const;

  executorch::runtime::Error Unload();

 private:
  static constexpr int required_num_providers_{1};

  const QnnInterface_t* StartBackend(
      const std::string& lib_path,
      const QnnSaver_Config_t** saver_config);

  executorch::runtime::Error InitBackend(
      void* const lib_handle,
      const QnnSaver_Config_t** saver_config);

  std::string lib_path_;
  void* lib_handle_{nullptr};
  QnnInterface qnn_interface_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
