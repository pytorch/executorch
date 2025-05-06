# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Enforce config names to always start with EXECUTORCH_, else raise an error.
function(enforce_executorch_config_name NAME)
  if(NOT "${NAME}" MATCHES "^EXECUTORCH_")
    message(FATAL_ERROR "Config name '${NAME}' must start with EXECUTORCH_")
  endif()
endfunction()

# Define an overridable config.
#   1) If the config is already defined in the process, then store that in cache
#   2) If the config is NOT set, then store the default value in cache
macro(define_overridable_config NAME DESCRIPTION VALUE_TYPE DEFAULT_VALUE)
  enforce_executorch_config_name(${NAME})

  if(NOT "${VALUE_TYPE}" STREQUAL "STRING" AND NOT "${VALUE_TYPE}" STREQUAL "BOOL")
    message(FATAL_ERROR "Invalid config (${NAME}) value type '${VALUE_TYPE}', must be either STRING or BOOL")
  endif()

  if(DEFINED ${NAME})
    set(${NAME} ${${NAME}} CACHE ${VALUE_TYPE} ${DESCRIPTION} FORCE)
  else()
    set(${NAME} ${DEFAULT_VALUE} CACHE ${VALUE_TYPE} ${DESCRIPTION})
  endif()
endmacro()
