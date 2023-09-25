/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_SYS_IMPLEMENTATION_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_SYS_IMPLEMENTATION_H_
#include <executorch/backends/qnn/runtime/Logging.h>
#include <executorch/backends/qnn/runtime/backends/QnnSysFunctionInterface.h>

#include <string>
namespace torch {
namespace executor {
namespace qnn {
class QnnSystemImplementation {
 public:
  explicit QnnSystemImplementation(std::string lib_path)
      : lib_path_(std::move(lib_path)){};

  Error Load();

  const QnnSystemInterface& GetQnnSystemInterface() const;

  Error Unload();

 private:
  static constexpr const int required_num_providers_{1};

  std::string lib_path_;
  QnnSystemInterface qnn_sys_interface_;
  void* lib_handle_{nullptr};
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch

#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_SYS_IMPLEMENTATION_H_
