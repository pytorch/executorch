/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "System/QnnSystemInterface.h"

#include <utility>

#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto qnn_##F(Args... args) const {                                  \
    return (qnn_sys_interface_->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }
namespace torch {
namespace executor {
namespace qnn {
using QnnSystemInterfaceGetProvidersFn =
    decltype(QnnSystemInterface_getProviders);

class QnnSystemInterface {
 public:
  QnnSystemInterface() = default;

  void SetQnnSystemInterface(const QnnSystemInterface_t* qnn_sys_interface) {
    qnn_sys_interface_ = qnn_sys_interface;
  }

  bool IsLoaded() const {
    return qnn_sys_interface_ != nullptr;
  }

  DEFINE_SHIM_FUNCTION_SYS_INTERFACE(
      system_context_create,
      systemContextCreate);
  DEFINE_SHIM_FUNCTION_SYS_INTERFACE(
      system_context_get_binary_info,
      systemContextGetBinaryInfo);
  DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

 private:
  const QnnSystemInterface_t* qnn_sys_interface_{nullptr};
};
} // namespace qnn
} // namespace executor
} // namespace torch
