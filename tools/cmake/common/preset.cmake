# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Announce the name and value of a cmake variable in the summary of the build.
function(announce_configured_options NAME)
  get_property(_options GLOBAL PROPERTY _announce_configured_options)
  if(NOT _options)
    set_property(GLOBAL PROPERTY _announce_configured_options)
    get_property(_options GLOBAL PROPERTY _announce_configured_options)
  endif()

  set(option_exists FALSE)
  foreach(_option IN LISTS _options)
    if(_option STREQUAL "${NAME}")
      set(option_exists TRUE)
      break()
    endif()
  endforeach()

  if(NOT option_exists)
    set(_options ${_options} "${NAME}")
    set_property(GLOBAL PROPERTY _announce_configured_options "${_options}")
  endif()
endfunction()

# Print the configured options.
function(print_configured_options)
  get_property(_options GLOBAL PROPERTY _announce_configured_options)
  list(SORT _options)

  set(_longest_name_length 0)
  foreach(_option IN LISTS _options)
    string(LENGTH "${_option}" length)
    if(length GREATER _longest_name_length)
      set(_longest_name_length ${length})
    endif()
  endforeach()

  message(STATUS "--- Configured Options ---\n")
  foreach(_option IN LISTS _options)
    string(LENGTH "${_option}" _option_length)
    math(EXPR num_spaces "${_longest_name_length} - ${_option_length}")
    set(padding "")
    while(num_spaces GREATER 0)
      set(padding "${padding} ")
      math(EXPR num_spaces "${num_spaces} - 1")
    endwhile()
    if(DEFINED ${_option})
      message(STATUS "${_option}${padding} : ${${_option}}")
    else()
      message(STATUS "${_option}${padding} x (unset)")
    endif()
  endforeach()
  message(STATUS "--------------------------")
endfunction()

# Enforce option names to always start with EXECUTORCH.
function(enforce_executorch_option_name NAME)
  if(NOT "${NAME}" MATCHES "^EXECUTORCH_")
    message(FATAL_ERROR "Option name '${NAME}' must start with EXECUTORCH_")
  endif()
endfunction()

# Define an overridable option. 1) If the option is already defined in the
# process, then store that in cache 2) If the option is NOT set, then store the
# default value in cache
macro(define_overridable_option NAME DESCRIPTION VALUE_TYPE DEFAULT_VALUE)
  enforce_executorch_option_name(${NAME})

  if(NOT "${VALUE_TYPE}" STREQUAL "STRING" AND NOT "${VALUE_TYPE}" STREQUAL
                                               "BOOL"
  )
    message(
      FATAL_ERROR
        "Invalid option (${NAME}) value type '${VALUE_TYPE}', must be either STRING or BOOL"
    )
  endif()

  if(DEFINED ${NAME} AND NOT DEFINED CACHE{${NAME}})
    set(${NAME}
        ${${NAME}}
        CACHE ${VALUE_TYPE} ${DESCRIPTION} FORCE
    )
  else()
    set(${NAME}
        ${DEFAULT_VALUE}
        CACHE ${VALUE_TYPE} ${DESCRIPTION}
    )
  endif()

  announce_configured_options(${NAME})
endmacro()

# Set an overridable option.
macro(set_overridable_option NAME VALUE)
  # If the user has explitily set the option, do not override it.
  if(NOT DEFINED ${NAME})
    set(${NAME}
        ${VALUE}
        CACHE STRING ""
    )
  endif()
endmacro()

# Detemine the build preset and load it.
macro(load_build_preset)
  if(DEFINED EXECUTORCH_BUILD_PRESET_FILE)
    announce_configured_options(EXECUTORCH_BUILD_PRESET_FILE)
    message(STATUS "Loading build preset: ${EXECUTORCH_BUILD_PRESET_FILE}")
    include(${EXECUTORCH_BUILD_PRESET_FILE})
  endif()
  # For now, just continue if the preset file is not set. In the future, we will
  # try to determine a preset file.
endmacro()

# Check if the required options are set.
function(check_required_options_on)
  cmake_parse_arguments(ARG "" "IF_ON" "REQUIRES" ${ARGN})

  if(${${ARG_IF_ON}})
    foreach(required ${ARG_REQUIRES})
      if(NOT ${${required}})
        message(FATAL_ERROR "Use of '${ARG_IF_ON}' requires '${required}'")
      endif()
    endforeach()
  endif()
endfunction()

# Check if flags conflict with each other.
function(check_conflicting_options_on)
  cmake_parse_arguments(ARG "" "IF_ON" "CONFLICTS_WITH" ${ARGN})

  if(${${ARG_IF_ON}})
    foreach(conflict ${ARG_CONFLICTS_WITH})
      if(${${conflict}})
        message(FATAL_ERROR "Both '${ARG_IF_ON}' and '${conflict}' can't be ON")
      endif()
    endforeach()
  endif()
endfunction()
