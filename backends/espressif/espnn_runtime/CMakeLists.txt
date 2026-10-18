# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

set(espnn_sources "src/xts3_graph.cpp" "src/int8_interpreter.cpp")
if(CONFIG_ESPNN_EXECUTORCH_DELEGATE)
  list(APPEND espnn_sources "src/espnn_backend.cpp")
  if(CONFIG_ESPNN_PROVIDE_EXECUTORCH_PAL)
    list(APPEND espnn_sources "src/espnn_pal.cpp")
  endif()
endif()

idf_component_register(
  SRCS
  ${espnn_sources}
  INCLUDE_DIRS
  "include"
  REQUIRES
  esp_timer
  heap
  freertos
  PRIV_REQUIRES
  esp-nn
  WHOLE_ARCHIVE
)

if(CONFIG_ESPNN_EXECUTORCH_DELEGATE)
  get_filename_component(
    default_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
  )
  set(ESPNN_EXECUTORCH_SOURCE_DIR
      "${default_root}"
      CACHE PATH "ExecuTorch source tree"
  )
  set(executorch_root "${ESPNN_EXECUTORCH_SOURCE_DIR}")
  set(ESPNN_EXECUTORCH_BUILD_DIR
      "${executorch_root}/cmake-out-esp-${IDF_TARGET}"
      CACHE
        PATH
        "ExecuTorch cross-compiled for this target (examples/espressif/scripts/build_executorch_esp.sh)"
  )
  set(et_lib "${ESPNN_EXECUTORCH_BUILD_DIR}/lib")
  foreach(required libexecutorch.a libexecutorch_core.a libportable_kernels.a
                   libkernels_util_all_deps.a
  )
    if(NOT EXISTS "${et_lib}/${required}")
      message(
        FATAL_ERROR
          "Missing ${et_lib}/${required}. Build ExecuTorch for ${IDF_TARGET} with "
          "examples/espressif/scripts/build_executorch_esp.sh or set "
          "-DESPNN_EXECUTORCH_BUILD_DIR=<dir>"
      )
    endif()
  endforeach()
  set(et_registries "")
  foreach(registry libportable_ops_lib.a libexecutorch_selected_kernels.a
                   libquantized_ops_lib.a
  )
    if(EXISTS "${et_lib}/${registry}")
      list(APPEND et_registries "${et_lib}/${registry}")
    endif()
  endforeach()
  set(et_kernels "${et_lib}/libportable_kernels.a")
  if(EXISTS "${et_lib}/libquantized_kernels.a")
    list(APPEND et_kernels "${et_lib}/libquantized_kernels.a")
  endif()
  target_include_directories(
    ${COMPONENT_LIB}
    PUBLIC
      "${ESPNN_EXECUTORCH_BUILD_DIR}/include"
      "${ESPNN_EXECUTORCH_BUILD_DIR}/include/executorch/runtime/core/portable_type/c10"
      "${executorch_root}/.."
  )
  target_compile_definitions(
    ${COMPONENT_LIB} PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
  )
  # Kernel and backend registration happens in static initializers, so these
  # archives are linked whole. One flag per archive: CMake deduplicates a bare
  # -Wl,--whole-archive pair.
  set(et_whole_archives "")
  foreach(archive ${et_registries} "${et_lib}/libexecutorch.a")
    list(APPEND et_whole_archives
         "-Wl,--whole-archive,${archive},--no-whole-archive"
    )
  endforeach()
  target_link_libraries(
    ${COMPONENT_LIB}
    INTERFACE ${et_whole_archives} ${et_kernels}
              "${et_lib}/libkernels_util_all_deps.a"
              "${et_lib}/libexecutorch_core.a"
  )
endif()
