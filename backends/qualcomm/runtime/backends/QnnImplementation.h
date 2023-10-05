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

#include <mutex>
#include <string>
#include <unordered_map>
namespace torch {
namespace executor {
namespace qnn {
class QnnImplementation {
 public:
  using BackendIdType = decltype(QnnInterface_t{}.backendId);

  explicit QnnImplementation(std::string lib_path)
      : lib_path_(std::move(lib_path)){};

  Error Load(const QnnSaver_Config_t** saver_config);

  const QnnInterface& GetQnnInterface() const;

  Error TerminateAllBackends();

 private:
  static constexpr const int required_num_providers_{1};

  static Error StartBackend(
      const std::string& lib_path,
      const QnnSaver_Config_t** saver_config);

  static Error InitBackend(
      void* const lib_handle,
      const QnnSaver_Config_t** saver_config);

  std::string lib_path_;
  QnnInterface qnn_interface_;

  static std::unordered_map<std::string, BackendIdType> lib_path_to_backend_id_;
  static std::unordered_map<BackendIdType, const QnnInterface_t*>
      loaded_backend_;
  static std::unordered_map<BackendIdType, void*> loaded_lib_handle_;
  static std::mutex be_init_mutex_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
