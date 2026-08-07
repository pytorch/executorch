# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(validate_gemma4_wasm_names export_variable output_variable)
  if(NOT DEFINED ${export_variable})
    message(FATAL_ERROR "${export_variable} must be defined")
  endif()
  if(NOT DEFINED ${output_variable})
    message(FATAL_ERROR "${output_variable} must be defined")
  endif()

  set(export_name "${${export_variable}}")
  set(output_name "${${output_variable}}")
  if(NOT export_name MATCHES "^[A-Za-z_$][A-Za-z0-9_$]*$")
    message(
      FATAL_ERROR
        "${export_variable} must be a JavaScript identifier: '${export_name}'"
    )
  endif()
  if(NOT output_name MATCHES "^[A-Za-z0-9][A-Za-z0-9._-]*$")
    message(
      FATAL_ERROR
        "${output_variable} must be a file-name stem: '${output_name}'"
    )
  endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE AND GEMMA4_VALIDATE_WASM_NAMES)
  validate_gemma4_wasm_names(
    GEMMA4_SPEC_WASM_EXPORT_NAME GEMMA4_SPEC_WASM_OUTPUT_NAME
  )
endif()
