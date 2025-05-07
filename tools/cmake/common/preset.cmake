# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Enforce option names to always start with EXECUTORCH.
function(enforce_executorch_option_name NAME)
  if(NOT "${NAME}" MATCHES "^EXECUTORCH_")
    message(FATAL_ERROR "Option name '${NAME}' must start with EXECUTORCH_")
  endif()
endfunction()

# Define an overridable option.
#   1) If the option is already defined in the process, then store that in cache
#   2) If the option is NOT set, then store the default value in cache
macro(define_overridable_option NAME DESCRIPTION VALUE_TYPE DEFAULT_VALUE)
  enforce_executorch_option_name(${NAME})

  if(NOT "${VALUE_TYPE}" STREQUAL "STRING" AND NOT "${VALUE_TYPE}" STREQUAL "BOOL")
    message(FATAL_ERROR "Invalid option (${NAME}) value type '${VALUE_TYPE}', must be either STRING or BOOL")
  endif()

  if(DEFINED ${NAME})
    set(${NAME} ${${NAME}} CACHE ${VALUE_TYPE} ${DESCRIPTION} FORCE)
  else()
    set(${NAME} ${DEFAULT_VALUE} CACHE ${VALUE_TYPE} ${DESCRIPTION})
  endif()
endmacro()
