# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# qnn_implementation
target_sources(
  qnn_implementation
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnImplementation.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnImplementation.cpp
)

# qnn_interface
target_sources(
  qnn_function_interface
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/QnnFunctionInterface.h
)

# qnn_sys_implementation
target_sources(
  qnn_sys_implementation
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnSysImplementation.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnSysImplementation.cpp
)

# qnn_sys_interface
target_sources(
  qnn_sys_function_interface
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/QnnSysFunctionInterface.h
)

# qnn_logger
target_sources(
  qnn_logger
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnLogger.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnLogger.cpp
)

# qnn_profiler
target_sources(
  qnn_profiler PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnProfiler.h
                       ${CMAKE_CURRENT_LIST_DIR}/QnnProfiler.cpp
)

set(HOST_ARCHITECTURE_GPU
    ${CMAKE_CURRENT_LIST_DIR}/gpu/${CMAKE_SYSTEM_PROCESSOR}
)
set(HOST_ARCHITECTURE_HTP
    ${CMAKE_CURRENT_LIST_DIR}/htp/${CMAKE_SYSTEM_PROCESSOR}
)
set(HOST_ARCHITECTURE_IR
    ${CMAKE_CURRENT_LIST_DIR}/ir/${CMAKE_SYSTEM_PROCESSOR}
)

# qnn_device
target_sources(
  qnn_device
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnDeviceCommon.h
         ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuDevice.h
         ${CMAKE_CURRENT_LIST_DIR}/htp/HtpDevice.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnDeviceCommon.cpp
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpDevice.cpp
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpDevicePlatformInfoConfig.h
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpDeviceCustomConfig.h
          # When offline prepare context cache in x86 host we have to provide
          # platform infomation and SocModel to Qnn
          ${HOST_ARCHITECTURE_HTP}/HtpDevicePlatformInfoConfig.cpp
          ${HOST_ARCHITECTURE_HTP}/HtpDeviceCustomConfig.cpp
)

# qnn_context
target_sources(
  qnn_context
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnContextCommon.h
         ${CMAKE_CURRENT_LIST_DIR}/htp/HtpContext.h
         ${CMAKE_CURRENT_LIST_DIR}/ir/IrContext.h
         ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuContext.h
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/QnnContextCommon.cpp
    ${CMAKE_CURRENT_LIST_DIR}/htp/HtpContext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/htp/HtpContextCustomConfig.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuContext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuContextCustomConfig.h
    ${HOST_ARCHITECTURE_GPU}/GpuContextCustomConfig.cpp
    ${HOST_ARCHITECTURE_HTP}/HtpContextCustomConfig.cpp
    ${HOST_ARCHITECTURE_IR}/IrContext.cpp
)

# qnn_backend_cache
target_sources(
  qnn_backend_cache
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnBackendCache.h
         ${CMAKE_CURRENT_LIST_DIR}/htp/HtpBackendCache.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnBackendCache.cpp
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpBackendCache.cpp
)

# qnn_graph
target_sources(
  qnn_graph
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnGraphCommon.h
         ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuGraph.h
         ${CMAKE_CURRENT_LIST_DIR}/htp/HtpGraph.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnGraphCommon.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuGraph.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuGraphCustomConfig.h
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuGraphCustomConfig.cpp
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpGraph.cpp
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpGraphCustomConfig.h
          ${CMAKE_CURRENT_LIST_DIR}/htp/HtpGraphCustomConfig.cpp
          ${HOST_ARCHITECTURE_HTP}/HtpGraphCustomConfig.cpp
)

# qnn_op_package_manager
target_sources(
  qnn_op_package_manager
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnOpPackageManager.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnOpPackageManager.cpp
)

# qnn_backend
target_sources(
  qnn_backend
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnBackendCommon.h
         ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuBackend.h
         ${CMAKE_CURRENT_LIST_DIR}/htp/HtpBackend.h
         ${CMAKE_CURRENT_LIST_DIR}/ir/IrBackend.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnBackendCommon.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuBackend.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuBackendCustomConfig.h
          ${CMAKE_CURRENT_LIST_DIR}/gpu/GpuBackendCustomConfig.cpp
)

# qnn_mem_manager
target_sources(
  qnn_mem_manager
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnMemManager.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnMemManager.cpp
)

# qnn_factory
target_sources(
  qnn_factory
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnBackendFactory.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnBackendFactory.cpp
)

# qnn_custom_protocol
target_sources(
  qnn_custom_protocol
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnCustomProtocol.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QnnCustomProtocol.cpp
)

# qnn_dlc_manager
target_sources(
  qnn_dlc_manager
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/QnnDlcManager.h
  PRIVATE
    ${HOST_ARCHITECTURE_IR}/QnnDlcManager.cpp
)
