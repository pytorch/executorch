# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Function to convert an absolute path to an identifier by making it relative
# to PROJECT_SOURCE_DIR and replacing slashes with underscores
function(absolute_path_to_identifier ABSOLUTE_PATH OUTPUT_VAR)
  # Get the relative path from PROJECT_SOURCE_DIR
  file(RELATIVE_PATH rel_path ${PROJECT_SOURCE_DIR} ${ABSOLUTE_PATH})
  # Replace folders into namespaces
  string(REPLACE "/" "." result "${rel_path}")
  string(REPLACE "\\" "." result "${result}")
  # Prefix all targets to indicate creation by ExecuTorch
  set(${OUTPUT_VAR} ${result} PARENT_SCOPE)
endfunction()

# Convert a target name to include the relative path from the project root.
macro(set_target_name TARGET_NAME OUTPUT_VAR)
  absolute_path_to_identifier("${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}" ${OUTPUT_VAR})
endmacro()

# Get the current source directory relative to the project root.
function(set_relative_current_source_dir OUTPUT_VAR)
  file(RELATIVE_PATH rel_path ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  set(${OUTPUT_VAR} "executorch/${rel_path}" PARENT_SCOPE)
endfunction()
