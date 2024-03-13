/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "QnnInterface.h"
#include "Saver/QnnSaver.h"

#include <utility>

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto qnn_##F(Args... args) const {                       \
    return (qnn_interface_->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }
namespace torch {
namespace executor {
namespace qnn {
using QnnInterfaceGetProvidersFn = decltype(QnnInterface_getProviders);
using QnnSaverInitializeFn = decltype(QnnSaver_initialize);

class QnnInterface {
  friend class QnnImplementation;

 public:
  QnnInterface() = default;

  // --------- QnnBackend ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      backend_register_op_package,
      backendRegisterOpPackage);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      backend_validate_op_config,
      backendValidateOpConfig);
  DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);
  // --------- QnnDevice ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      device_get_infrastructure,
      deviceGetInfrastructure);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      device_get_platform_info,
      deviceGetPlatformInfo);
  // DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);
  // --------- QnnContext ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);
  DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      context_create_from_binary,
      contextCreateFromBinary);
  DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);
  // --------- QnnGraph ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);
  DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);
  DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);
  DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);
  // --------- QnnLog ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);
  DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);
  // --------- QnnProfile ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);
  DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);
  DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);
  DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);
  DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);
  // --------- QnnMem ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);
  DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);
  // --------- QnnProperty --------
  DEFINE_SHIM_FUNCTION_INTERFACE(
      property_has_capability,
      propertyHasCapability);
  // --------- QnnTensor ---------
  DEFINE_SHIM_FUNCTION_INTERFACE(
      tensor_create_context_tensor,
      tensorCreateContextTensor);
  DEFINE_SHIM_FUNCTION_INTERFACE(
      tensor_create_graph_tensor,
      tensorCreateGraphTensor);

  void SetQnnInterface(const QnnInterface_t* qnn_interface) {
    qnn_interface_ = qnn_interface;
  }

  uint32_t GetBackendId() const {
    return qnn_interface_->backendId;
  }

  bool IsLoaded() const {
    return qnn_interface_ != nullptr;
  }

 private:
  // --------- QnnInterface ---------
  const QnnInterface_t* qnn_interface_{nullptr};
};
} // namespace qnn
} // namespace executor
} // namespace torch
