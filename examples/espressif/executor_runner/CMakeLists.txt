# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ESP-IDF component CMakeLists.txt for the ExecuTorch executor runner.
#
# This file defines the executor_runner as an ESP-IDF component. It is designed
# to work with the ESP-IDF build system (idf.py build).
#
# Project structure expected: my_project/ ├── CMakeLists.txt (project-level,
# uses this as a component) ├── main/ │   └── CMakeLists.txt (main component,
# depends on executor_runner) └── components/ └── executor_runner/     (this
# component - symlink or copy)
#
# Or you can use this CMakeLists.txt directly as a standalone CMake build for
# cross-compilation testing.

cmake_minimum_required(VERSION 3.16)

# ─── Option: ESP-IDF component mode vs. standalone CMake mode ───
if(ESP_PLATFORM)
  # ═══════════════════════════════════════════════════════════════
  # ESP-IDF Component Build
  # ═══════════════════════════════════════════════════════════════
  idf_component_register(
    SRCS
    "esp_executor_runner.cpp"
    "esp_pal.cpp"
    "esp_memory_allocator.cpp"
    "esp_perf_monitor.cpp"
    INCLUDE_DIRS
    "."
    REQUIRES
    esp_timer
    esp_system
    spiffs
  )

  # ExecuTorch pre-built library paths
  set(ET_DIR_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
      CACHE PATH "Path to ExecuTorch source dir"
  )
  set(ET_BUILD_DIR_PATH
      "${ET_DIR_PATH}/cmake-out-esp"
      CACHE PATH "Path to ExecuTorch build/install dir for ESP target"
  )
  set(ET_PTE_FILE_PATH
      ""
      CACHE PATH "Path to ExecuTorch model .pte file"
  )
  set(PYTHON_EXECUTABLE
      "python3"
      CACHE PATH "Python executable"
  )

  set(ET_NUM_INFERENCES
      "10"
      CACHE STRING "Number of inferences to run"
  )
  option(ET_LOG_DUMP_INPUT "Dump input in log" OFF)
  option(ET_LOG_DUMP_OUTPUT "Dump output in log" ON)
  option(ET_BUNDLE_IO "Set to compile in BundleIO support" OFF)
  set(ET_ATOL
      "0.01"
      CACHE STRING "Absolute tolerance for BundleIO testing"
  )
  set(ET_RTOL
      "0.01"
      CACHE STRING "Relative tolerance for BundleIO testing"
  )
  option(ET_DUMP_OUTPUTS "Collect and print outputs as base64 in log" OFF)
  option(ET_DUMP_INTERMEDIATE_OUTPUTS "Collect and print intermediate outputs"
         OFF
  )
  set(ET_DEBUG_BUFFER_SIZE
      "65536"
      CACHE STRING "Size of ETDump debug buffer"
  )
  option(FILESYSTEM_LOAD
         "Load model from filesystem instead of compiled-in data" OFF
  )

  # Directory containing the generated model_pte.h header. By default this is
  # the project source directory (where build.sh places it), but it can be
  # overridden if you generate the header elsewhere.
  set(ET_MODEL_HEADER_DIR
      "${CMAKE_SOURCE_DIR}"
      CACHE PATH "Directory containing the generated model_pte.h header"
  )

  # Memory pool sizes
  set(ET_ESP_METHOD_ALLOCATOR_POOL_SIZE
      ""
      CACHE
        STRING
        "Method allocator pool size (empty = auto based on PSRAM availability)"
  )
  set(ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
      ""
      CACHE
        STRING
        "Scratch temp allocator pool size (empty = auto based on PSRAM availability)"
  )

  # Find pre-built ExecuTorch libraries. TARGETS_GLOBAL is needed because
  # ESP-IDF's project.cmake resolves link dependencies from the top-level
  # project scope, but find_package runs inside this component's directory
  # scope. Without GLOBAL, the imported targets (executorch, portable_kernels,
  # etc.) are invisible at the project level and you get "No target executorch"
  # errors.
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
  find_package(
    executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
  )

  # Convert pte to header if not using filesystem loading
  if(NOT FILESYSTEM_LOAD AND ET_PTE_FILE_PATH)
    add_custom_target(
      gen_model_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
    )
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pte_to_header.py
              --pte ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${ET_PTE_FILE_PATH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(${COMPONENT_LIB} gen_model_header)
  endif()

  # Include directories
  target_include_directories(
    ${COMPONENT_LIB}
    PRIVATE ${ET_DIR_PATH}/.. ${ET_DIR_PATH}/runtime/core/portable_type/c10
            ${CMAKE_CURRENT_BINARY_DIR} ${ET_MODEL_HEADER_DIR}
  )

  # Link ExecuTorch libraries
  set(esp_runner_libs)
  list(APPEND esp_runner_libs extension_runner_util executorch
       executorch_selected_kernels
  )

  if(TARGET xnnpack_backend)
    list(APPEND esp_runner_libs xnnpack_backend)
  endif()

  if(EXECUTORCH_ENABLE_EVENT_TRACER)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_EVENT_TRACER_ENABLED)
    list(APPEND esp_runner_libs etdump flatccrt)
  endif()

  if(ET_BUNDLE_IO)
    list(APPEND esp_runner_libs bundled_program)
  endif()

  target_link_libraries(${COMPONENT_LIB} PUBLIC ${esp_runner_libs})

  # Compile definitions
  target_compile_definitions(
    ${COMPONENT_LIB} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
  )

  if(ET_NUM_INFERENCES)
    target_compile_definitions(
      ${COMPONENT_LIB} PUBLIC ET_NUM_INFERENCES=${ET_NUM_INFERENCES}
    )
  endif()

  if(ET_LOG_DUMP_INPUT)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_LOG_DUMP_INPUT)
  endif()

  if(ET_LOG_DUMP_OUTPUT)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_LOG_DUMP_OUTPUT)
  endif()

  if(ET_BUNDLE_IO)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_BUNDLE_IO)
  endif()

  if(ET_ATOL)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_ATOL=${ET_ATOL})
  endif()

  if(ET_RTOL)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_RTOL=${ET_RTOL})
  endif()

  if(ET_DUMP_OUTPUTS)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ET_DUMP_OUTPUTS)
  endif()

  if(ET_DUMP_INTERMEDIATE_OUTPUTS)
    target_compile_definitions(
      ${COMPONENT_LIB} PUBLIC ET_DUMP_INTERMEDIATE_OUTPUTS
    )
  endif()

  if(ET_DEBUG_BUFFER_SIZE)
    target_compile_definitions(
      ${COMPONENT_LIB} PUBLIC ET_DEBUG_BUFFER_SIZE=${ET_DEBUG_BUFFER_SIZE}
    )
  endif()

  if(FILESYSTEM_LOAD)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC FILESYSTEM_LOAD)
  endif()

  if(ET_ESP_METHOD_ALLOCATOR_POOL_SIZE)
    target_compile_definitions(
      ${COMPONENT_LIB}
      PUBLIC
        ET_ESP_METHOD_ALLOCATOR_POOL_SIZE=${ET_ESP_METHOD_ALLOCATOR_POOL_SIZE}
    )
  endif()

  if(ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    target_compile_definitions(
      ${COMPONENT_LIB}
      PUBLIC
        ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
    )
  endif()

else()
  # ═══════════════════════════════════════════════════════════════
  # Standalone CMake Build (for host testing / cross-compilation)
  # ═══════════════════════════════════════════════════════════════
  project(esp_executor_runner)

  set(ET_DIR_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
      CACHE PATH "Path to ExecuTorch dir"
  )
  include(${ET_DIR_PATH}/tools/cmake/Utils.cmake)
  set(ET_BUILD_DIR_PATH
      "${ET_DIR_PATH}/cmake-out"
      CACHE PATH "Path to ExecuTorch build/install dir"
  )
  set(ET_INCLUDE_PATH
      "${ET_DIR_PATH}/.."
      CACHE PATH "Path to ExecuTorch headers"
  )
  set(ET_PTE_FILE_PATH
      ""
      CACHE PATH "Path to ExecuTorch model pte"
  )
  set(PYTHON_EXECUTABLE
      "python3"
      CACHE PATH "Python executable"
  )

  set(ET_NUM_INFERENCES
      "1"
      CACHE STRING "Number of inferences to run"
  )
  option(ET_LOG_DUMP_OUTPUT "Dump output in log" ON)

  if(NOT DEFINED ET_PTE_FILE_PATH OR ET_PTE_FILE_PATH STREQUAL "")
    message(FATAL_ERROR "ET_PTE_FILE_PATH must be set to the .pte model file")
  endif()

  find_package(
    executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
  )

  # Convert pte to header
  add_custom_target(
    gen_model_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/pte_to_header.py --pte
            ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ET_PTE_FILE_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_executable(esp_executor_runner)
  target_sources(
    esp_executor_runner PRIVATE esp_executor_runner.cpp esp_pal.cpp
                                esp_perf_monitor.cpp esp_memory_allocator.cpp
  )

  target_link_libraries(
    esp_executor_runner PUBLIC extension_runner_util executorch
                               portable_kernels
  )

  target_include_directories(
    esp_executor_runner
    PRIVATE ${ET_INCLUDE_PATH} ${ET_DIR_PATH}/runtime/core/portable_type/c10
            ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(
    esp_executor_runner PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
  )

  if(ET_NUM_INFERENCES)
    target_compile_definitions(
      esp_executor_runner PUBLIC ET_NUM_INFERENCES=${ET_NUM_INFERENCES}
    )
  endif()

  if(ET_LOG_DUMP_OUTPUT)
    target_compile_definitions(esp_executor_runner PUBLIC ET_LOG_DUMP_OUTPUT)
  endif()

  add_dependencies(esp_executor_runner gen_model_header)
endif()
