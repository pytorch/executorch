# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Resolve dependencies on the consuming machine, including C++-only projects.
function(executorch_coreai_dependencies)
  if(TARGET executorch::coreai_dependencies)
    return()
  endif()
  if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|iOS)$")
    message(FATAL_ERROR "Core AI requires macOS or iOS")
  endif()

  set(sdk "${CMAKE_OSX_SYSROOT}")
  if(NOT sdk)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
      set(sdk iphoneos)
    else()
      set(sdk macosx)
    endif()
  endif()
  execute_process(
    COMMAND xcrun --sdk "${sdk}" --show-sdk-version
    OUTPUT_VARIABLE sdk_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE sdk_result
    ERROR_VARIABLE sdk_error
  )
  if(NOT sdk_result EQUAL 0 OR sdk_version VERSION_LESS 27.0)
    message(
      FATAL_ERROR
        "Core AI requires an Apple SDK 27 or newer; selected '${sdk}' "
        "reports '${sdk_version}'. Select Xcode 27 with DEVELOPER_DIR. "
        "${sdk_error}"
    )
  endif()
  execute_process(
    COMMAND xcrun --sdk "${sdk}" --show-sdk-path
    OUTPUT_VARIABLE sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(
    foundation Foundation
    PATHS "${sdk_path}/System/Library/Frameworks"
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH NO_CACHE REQUIRED
  )
  find_library(
    coreai CoreAI
    PATHS "${sdk_path}/System/Library/Frameworks"
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH NO_CACHE REQUIRED
  )

  # C++ package consumers need Swift's autolink paths, not the Swift language.
  set(swift_compiler "${CMAKE_Swift_COMPILER}")
  if(NOT swift_compiler)
    execute_process(
      COMMAND xcrun --sdk "${sdk}" --find swiftc
      OUTPUT_VARIABLE swift_compiler
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
  set(arch "${CMAKE_OSX_ARCHITECTURES}")
  if(NOT arch)
    set(arch "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT arch)
    set(arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  endif()
  list(GET arch 0 arch)
  set(deployment 27.0)
  if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER deployment)
    set(deployment "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(target_environment "")
  if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(platform ios)
    if(sdk_path MATCHES "[Ss]imulator")
      set(target_environment -simulator)
    endif()
  else()
    set(platform macosx)
  endif()
  execute_process(
    COMMAND "${swift_compiler}" -print-target-info -sdk "${sdk_path}" -target
            "${arch}-apple-${platform}${deployment}${target_environment}"
    OUTPUT_VARIABLE target_info COMMAND_ERROR_IS_FATAL ANY
  )
  string(JSON path_count LENGTH "${target_info}" paths runtimeLibraryPaths)
  set(runtime_paths "")
  if(path_count GREATER 0)
    math(EXPR path_last "${path_count} - 1")
    foreach(index RANGE ${path_last})
      string(
        JSON
        runtime_path
        GET
        "${target_info}"
        paths
        runtimeLibraryPaths
        ${index}
      )
      list(APPEND runtime_paths "${runtime_path}")
    endforeach()
  endif()

  # GLOBAL keeps the target visible to runners outside the backend directory.
  add_library(executorch::coreai_dependencies INTERFACE IMPORTED GLOBAL)
  set_target_properties(
    executorch::coreai_dependencies
    PROPERTIES
      INTERFACE_LINK_LIBRARIES "${foundation};${coreai}"
      INTERFACE_LINK_DIRECTORIES "${runtime_paths}"
      INTERFACE_LINK_OPTIONS
      "SHELL:-F \"${sdk_path}/System/Library/SubFrameworks\";LINKER:-rpath,/usr/lib/swift"
  )
endfunction()
