/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <unordered_set>

namespace torch {
namespace executor {
namespace qnn {

class QnnMemManager {
 public:
  explicit QnnMemManager(
      const QnnImplementation& implementation,
      QnnContext* context)
      : implementation_(implementation), context_(context) {}
  ~QnnMemManager() {
    DeRegisterMem();
  }

  Error RegisterMem(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      int32_t mem_fd);

  bool IsRegistered(Qnn_MemHandle_t handle);

 private:
  void DeRegisterMem();

  const QnnImplementation& implementation_;
  QnnContext* context_;
  std::unordered_set<Qnn_MemHandle_t> registered_set_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
