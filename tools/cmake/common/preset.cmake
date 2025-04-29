# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Announce the name and value of a cmake variable in the summary of the build.
function(announce_configured_options NAME)
  get_property(configs GLOBAL PROPERTY _announce_configured_options)
  if(NOT configs)
    set_property(GLOBAL PROPERTY _announce_configured_options)
    get_property(configs GLOBAL PROPERTY _announce_configured_options)
  endif()

  set(option_exists FALSE)
  foreach(config IN LISTS configs)
    if(config STREQUAL "${NAME}")
      set(option_exists TRUE)
      break()
    endif()
  endforeach()

  if(NOT option_exists)
    set(configs ${configs} "${NAME}")
    set_property(GLOBAL PROPERTY _announce_configured_options "${configs}")
  endif()
endfunction()

# Print the configured options.
function(print_configured_options)
  get_property(configs GLOBAL PROPERTY _announce_configured_options)

  set(longest_name_length 0)
  foreach(config IN LISTS configs)
    string(LENGTH "${config}" length)
    if(length GREATER longest_name_length)
      set(longest_name_length ${length})
    endif()
  endforeach()

  message(STATUS "--- ⚙️ Configurations ---\n")
  foreach(config IN LISTS configs)
    string(LENGTH "${config}" config_length)
    math(EXPR num_spaces "${longest_name_length} - ${config_length}")
    set(padding "")
    while(num_spaces GREATER 0)
      set(padding "${padding} ")
      math(EXPR num_spaces "${num_spaces} - 1")
    endwhile()
    message(STATUS "${config}${padding} : ${${config}}")
  endforeach()
  message(STATUS "------------------------")
endfunction()

# Detemine the build preset and load it.
macro(determine_and_load_build_preset)
  if(NOT EXECUTORCH_BUILD_PRESET_FILE)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(EXECUTORCH_BUILD_PRESET_FILE "${PROJECT_SOURCE_DIR}/tools/cmake/preset/macos-arm64.cmake" CACHE STRING "Build preset" FORCE)
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(EXECUTORCH_BUILD_PRESET_FILE "${PROJECT_SOURCE_DIR}/tools/cmake/preset/linux-x86_64.cmake" CACHE STRING "Build preset" FORCE)
      endif()
    endif()
  endif()

  if(EXECUTORCH_BUILD_PRESET_FILE)
    announce_configured_options(EXECUTORCH_BUILD_PRESET_FILE)
    message(STATUS "Loading build preset: ${EXECUTORCH_BUILD_PRESET_FILE}")
    include(${EXECUTORCH_BUILD_PRESET_FILE})
  else()
    message(WARNING "Unable to determine build preset with CMAKE_SYSTEM_NAME (${CMAKE_SYSTEM_NAME}) and CMAKE_SYSTEM_PROCESSOR (${CMAKE_SYSTEM_PROCESSOR}). Using default build settings.")
  endif()
endmacro()

function(_enforce_executorch_config_name NAME)
  if(NOT "${NAME}" MATCHES "^EXECUTORCH_")
    message(FATAL_ERROR "Config name '${NAME}' must start with EXECUTORCH_")
  endif()
endfunction()

# Do not define a config outside of this file.
macro(define_overridable_config NAME DESCRIPTION DEFAULT_VALUE)
  _enforce_executorch_config_name(${NAME})

  if(DEFINED ${NAME})
    set(${NAME} ${${NAME}} CACHE STRING ${DESCRIPTION} FORCE)
  else()
    set(${NAME} ${DEFAULT_VALUE} CACHE STRING ${DESCRIPTION})
  endif()

  announce_configured_options(${NAME})
endmacro()

# Set an overridable config.
macro(set_overridable_config NAME VALUE)
  _enforce_executorch_config_name(${NAME})

  # If the user has explitily set the config, do not override it.
  if(DEFINED ${NAME})
    return()
  endif()

  set(${NAME} ${VALUE} CACHE STRING "")
endmacro()
