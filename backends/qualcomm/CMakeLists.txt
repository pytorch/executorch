# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# get path
#
get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)
get_filename_component(
  QNN_EXECUTORCH_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE
)
# Let files say "include <executorch/path/to/header.h>".
get_filename_component(
  _common_include_directories "${EXECUTORCH_SOURCE_DIR}/.." ABSOLUTE
)

if(NOT DEFINED QNN_SDK_ROOT)
  message(
    FATAL_ERROR
      "Please define QNN_SDK_ROOT, e.g. cmake <..> -DQNN_SDK_ROOT=<...>"
  )
elseif(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  message(FATAL_ERROR "ios is not supported by Qualcomm AI Engine Direct")
endif()

message(STATUS "Using qnn sdk root ${QNN_SDK_ROOT}")
message(STATUS "Using EXECUTORCH_SOURCE_DIR ${EXECUTORCH_SOURCE_DIR}")

if(${ANDROID})
  find_library(android_log log)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

set(qcir_schema_include_dir ${CMAKE_CURRENT_LIST_DIR}/aot/ir)
set(qcir_schema_output ${qcir_schema_include_dir}/qcir_generated.h)
add_custom_command(
  OUTPUT qcir_schema_output
  COMMAND ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
          ${qcir_schema_include_dir} ${qcir_schema_include_dir}/qcir.fbs
  COMMENT "Generating qualcomm ir schema headers"
  VERBATIM
)

add_compile_options("-Wall" "-Werror" "-Wno-sign-compare")
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# GNU emit wanring for ignored attributes Unfortunately, we use [[maybe_unused]]
# which can be ignored by GNU. So we make it a warning, not an error in GNU.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wno-error=attributes")
  add_link_options("-flto=auto")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  # strip symbols
  add_link_options("-s")

  # --gc-sections is added by torch.
  add_compile_options("-O3" "-ffunction-sections" "-fdata-sections" "-frtti")
endif()

include_directories(
  BEFORE ${_common_include_directories} ${QNN_SDK_ROOT}/include/QNN
  ${EXECUTORCH_SOURCE_DIR}/third-party/flatbuffers/include
  ${EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type
)

set(_qnn_schema__srcs
  backends/qualcomm/serialization/qc_compiler_spec.fbs
  backends/qualcomm/serialization/qc_binary_info.fbs
)
set(_qnn_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")
# Paths to headers generated from the .fbs files.
set(_qnn_schema__outputs)
foreach(fbs_file ${_qnn_schema__srcs})
  string(REGEX REPLACE "serialization/([^/]+)[.]fbs$" "\\1_generated.h"
                       generated "${fbs_file}"
  )
  list(APPEND _qnn_schema__outputs
       "${_qnn_schema__include_dir}/executorch/${generated}"
  )
endforeach()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_qnn_schema__outputs}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${_qnn_schema__include_dir}/executorch/backends/qualcomm"
    ${_qnn_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
  COMMENT "Generating qnn_schema headers"
  VERBATIM
)

include_directories(
  BEFORE ${_qnn_schema__include_dir}
  ${EXECUTORCH_SOURCE_DIR}/third-party/flatbuffers/include
)

#
# declare targets
#
add_library(executorch_backend INTERFACE)
add_library(qcir INTERFACE qcir_schema_output)
add_library(qcir_utils STATIC)
add_library(qnn_backend STATIC)
add_library(qnn_backend_cache STATIC)
add_library(qnn_context STATIC)
add_library(qnn_device STATIC)
add_library(qnn_executorch_backend SHARED)
add_library(qnn_executorch_header INTERFACE)
add_library(qnn_executorch_logging STATIC)
add_library(qnn_factory STATIC)
add_library(qnn_function_interface INTERFACE)
add_library(qnn_graph STATIC)
add_library(qnn_header INTERFACE)
add_library(qnn_implementation STATIC)
add_library(qnn_logger STATIC)
add_library(qnn_manager STATIC)
add_library(qnn_mem_manager STATIC)
add_library(qnn_profiler STATIC)
add_library(qnn_schema INTERFACE ${_qnn_schema__outputs})
add_library(qnn_sys_function_interface INTERFACE)
add_library(qnn_sys_implementation STATIC)
add_library(shared_buffer STATIC)
add_library(wrappers STATIC)
add_library(utils STATIC)

#
# declare dependency
#
target_link_libraries(qcir_utils PRIVATE qcir)
target_link_libraries(wrappers PRIVATE qnn_header qnn_executorch_logging)
target_link_libraries(qnn_function_interface INTERFACE qnn_header)
target_link_libraries(
  qnn_implementation PRIVATE qnn_function_interface qnn_header
                             qnn_executorch_logging ${CMAKE_DL_LIBS}
)
target_link_libraries(qnn_sys_function_interface INTERFACE qnn_header)
target_link_libraries(
  qnn_sys_implementation PRIVATE qnn_sys_function_interface qnn_header
                                 qnn_executorch_logging ${CMAKE_DL_LIBS}
)
target_link_libraries(qnn_executorch_logging PRIVATE qnn_schema)
target_link_libraries(qnn_profiler PRIVATE qnn_executorch_logging)
target_link_libraries(qnn_logger PRIVATE qnn_implementation ${android_log})
target_link_libraries(qnn_backend PRIVATE qnn_implementation qnn_logger)
target_link_libraries(
  qnn_device PRIVATE qnn_executorch_logging qnn_implementation qnn_logger
)
target_link_libraries(
  qnn_backend_cache PRIVATE qnn_sys_implementation qcir_utils
)
target_link_libraries(
  qnn_context PRIVATE qnn_implementation qnn_logger qnn_backend qnn_device
                      qnn_backend_cache
)
target_link_libraries(
  qnn_graph PRIVATE qnn_executorch_logging qnn_implementation qnn_context
                    qnn_profiler
)
target_link_libraries(
  qnn_mem_manager PRIVATE qnn_executorch_logging qnn_implementation qnn_context
)

target_link_libraries(
  qnn_factory
  PUBLIC qnn_header
  PRIVATE qnn_schema qnn_backend qnn_device qnn_context qnn_graph
          qnn_mem_manager
)
target_link_libraries(
  qnn_manager PRIVATE qnn_factory wrappers qnn_schema utils shared_buffer
)
target_link_libraries(
  qnn_executorch_backend PRIVATE qnn_executorch_header qnn_schema qnn_manager
                                 executorch_core qcir_utils extension_tensor
)
set_target_properties(
  qnn_executorch_backend PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
target_link_libraries(utils PRIVATE qnn_executorch_logging)
target_link_libraries(
  shared_buffer PRIVATE qnn_executorch_logging ${CMAKE_DL_LIBS}
)
#
# add linker option
#
target_link_options_shared_lib(qnn_executorch_backend)

#
# add compile option
#
target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)

#
# add sources
#
add_subdirectory(
  ${QNN_EXECUTORCH_ROOT_DIR}/runtime ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch
)
add_subdirectory(
  ${QNN_EXECUTORCH_ROOT_DIR}/runtime/backends
  ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch/backends
)
add_subdirectory(
  ${QNN_EXECUTORCH_ROOT_DIR}/aot/wrappers
  ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch/wrappers
)
add_subdirectory(
  ${QNN_EXECUTORCH_ROOT_DIR}/aot/ir
  ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch/ir
)
install(TARGETS qnn_executorch_backend DESTINATION lib)

# QNN pybind
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  add_subdirectory(
    ${EXECUTORCH_SOURCE_DIR}/third-party/pybind11
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11
  )
  add_library(PyQnnManagerAdaptor MODULE)
  add_library(PyQnnWrapperAdaptor MODULE)
  # PyQnnManager containing a pybind type triggers the warning because pybind11
  # code internally forces hidden visibility.
  set_target_properties(
    PyQnnManagerAdaptor PROPERTIES CXX_VISIBILITY_PRESET hidden
  )
  set_target_properties(
    PyQnnWrapperAdaptor PROPERTIES CXX_VISIBILITY_PRESET hidden
  )

  target_link_libraries(
    PyQnnManagerAdaptor
    PRIVATE pybind11::module
            pybind11::lto
            qnn_schema
            qnn_manager
            qnn_executorch_header
            executorch
            qcir_utils
            extension_tensor
  )
  target_link_libraries(
    PyQnnWrapperAdaptor PRIVATE pybind11::module pybind11::lto wrappers
                                qnn_executorch_logging qnn_executorch_header
  )

  pybind11_extension(PyQnnManagerAdaptor)
  pybind11_extension(PyQnnWrapperAdaptor)
  if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary
    pybind11_strip(PyQnnManagerAdaptor)
    pybind11_strip(PyQnnWrapperAdaptor)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # need to allow exceptions in pybind
    set(_pybind_compile_options -Wno-deprecated-declarations -fPIC -frtti
                                -fexceptions
    )
    target_compile_options(
      PyQnnManagerAdaptor PUBLIC ${_pybind_compile_options}
    )
    target_compile_options(
      PyQnnWrapperAdaptor PUBLIC ${_pybind_compile_options}
    )
  endif()

  add_subdirectory(
    ${QNN_EXECUTORCH_ROOT_DIR}/aot/python
    ${CMAKE_CURRENT_BINARY_DIR}/qnn_executorch/python
  )
endif()
