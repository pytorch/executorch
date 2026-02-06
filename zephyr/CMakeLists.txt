# Copyright (c) 2025 Petri Oksanen
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_EXECUTORCH)

  set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

  set(EXECUTORCH_DIR ${ZEPHYR_CURRENT_MODULE_DIR})
  message(STATUS "EXECUTORCH_DIR set to: ${EXECUTORCH_DIR}")
  include(${EXECUTORCH_DIR}/tools/cmake/common/preset.cmake)
  include(${EXECUTORCH_DIR}/tools/cmake/preset/zephyr.cmake)

  # HACK: Force the parent directory of executorch into the global include path.
  # This is needed because the executorch CMake scripts generate a relative
  # include path ("../") that doesn't resolve correctly in a Zephyr out-of-tree
  # module build.
  get_filename_component(EXECUTORCH_PARENT_DIR ${EXECUTORCH_DIR} DIRECTORY)
  include_directories(${EXECUTORCH_PARENT_DIR})

  # Note: We don't create executorch_pal as a separate library to avoid circular
  # dependencies with Zephyr's build system. Instead, we'll add the PAL source
  # directly to our final library.
  set(EXECUTORCH_PAL_DEFAULT "zephyr")

  set(EXECUTORCH_BUILD_EXECUTORCH_PYBIND OFF)
  set(EXECUTORCH_BUILD_GENERATED_SRC OFF)
  set(EXECUTORCH_BUILD_SDK OFF)

  if(CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS)
    set(EXECUTORCH_BUILD_PORTABLE_OPS ON)
  else()
    set(EXECUTORCH_BUILD_PORTABLE_OPS OFF)
  endif()
  set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
  set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)

  # Backend selection
  if(CONFIG_CPU_CORTEX_M)
    set(EXECUTORCH_BUILD_CORTEX_M ON)
  endif()
  if(CONFIG_CPU_CORTEX_A)
    set(EXECUTORCH_BUILD_XNNPACK ON)
  endif()
  if(CONFIG_ETHOS_U)
    set(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
  endif()

  set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON)

  set(EXECUTORCH_ENABLE_LOGGING ${CONFIG_EXECUTORCH_ENABLE_LOGGING})
  set(EXECUTORCH_ENABLE_EVENT_TRACER ${CONFIG_EXECUTORCH_ENABLE_EVENT_TRACER})
  set(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION
      ${CONFIG_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION}
  )
  set(EXECUTORCH_OPTIMIZE_SIZE ${CONFIG_EXECUTORCH_OPTIMIZE_FOR_SIZE})

  add_subdirectory(${EXECUTORCH_DIR} ${CMAKE_CURRENT_BINARY_DIR}/executorch)

  # Get Zephyr's compiler flags that include the correct ARM architecture
  # settings
  get_property(
    ZEPHYR_COMPILE_OPTIONS
    TARGET zephyr_interface
    PROPERTY INTERFACE_COMPILE_OPTIONS
  )

  # Build with ZephyrOS compiler options
  set(BUILD_WITH_ZEPHYR_COMPILE_OPTIONS
      $<BUILD_INTERFACE:${ZEPHYR_COMPILE_OPTIONS}>
  )

  # Function to aggressively clean and fix target compile options
  function(fix_executorch_target_flags target_name)
    if(TARGET ${target_name})
      # Check if this is an INTERFACE target
      get_target_property(TARGET_TYPE ${target_name} TYPE)

      if(TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")

        # Apply corrected options as INTERFACE
        target_compile_options(
          ${target_name} INTERFACE ${BUILD_WITH_ZEPHYR_COMPILE_OPTIONS}
        )
        message(
          STATUS "Fixed INTERFACE compile flags for target: ${target_name}"
        )
      else()
        # Apply our corrected options
        target_compile_options(
          ${target_name} PUBLIC ${BUILD_WITH_ZEPHYR_COMPILE_OPTIONS}
        )
        message(STATUS "Fixed compile flags for target: ${target_name}")
      endif()
    endif()
  endfunction()

  # List of all ExecuTorch targets that need flag correction
  set(EXECUTORCH_TARGETS_TO_FIX
      executorch_core
      executorch
      portable_ops_lib
      portable_kernels
      optimized_portable_kernels
      quantized_kernels
      extension_data_loader
      extension_runner_util
      extension_tensor
      extension_flat_tensor
      extension_threadpool
      program_schema
      kernels_util_all_deps
      optimized_kernels
      cortex_m_ops_lib
      cortex_m_kernels
      cmsis-nn
  )

  # Apply corrected flags to each target
  foreach(target IN LISTS EXECUTORCH_TARGETS_TO_FIX)
    if(TARGET ${target})
      fix_executorch_target_flags(${target})
    else()
      message(STATUS "Target not found in explicit list: ${target}")
    endif()
  endforeach()

  if(TARGET zephyr_interface)
    target_link_libraries(executorch_core PUBLIC zephyr_interface)
  endif()

  # Create a single Zephyr library that includes ExecuTorch integration and PAL
  zephyr_library_named(libexecutorch)

  # Link ExecuTorch libraries
  target_link_libraries(libexecutorch PUBLIC executorch_core)

  # Only link portable_ops_lib if it was built (i.e., if portable ops are
  # enabled)
  if(TARGET portable_ops_lib)
    target_link_libraries(libexecutorch PUBLIC portable_ops_lib)
    message(STATUS "Linking with default portable_ops_lib")
  else()
    message(
      STATUS
        "portable_ops_lib not available - using selective operators from application"
    )
  endif()

  target_include_directories(
    libexecutorch PUBLIC ${EXECUTORCH_DIR}/runtime
                         ${EXECUTORCH_DIR}/third-party/flatcc/include
  )

endif()
