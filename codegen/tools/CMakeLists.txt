# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check if pybind11 is available

# Create the selective_build pybind11 module
pybind11_add_module(selective_build SHARED selective_build.cpp)

# Set the output name to match the module name
set_target_properties(selective_build PROPERTIES OUTPUT_NAME "selective_build")

# Set the module name for the pybind11 module
target_compile_definitions(
  selective_build PUBLIC EXECUTORCH_PYTHON_MODULE_NAME=selective_build
)

# Include directories
target_include_directories(
  selective_build PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Compile options
target_compile_options(
  selective_build PUBLIC -Wno-deprecated-declarations -fPIC -frtti -fexceptions
)

# Link against required libraries
target_link_libraries(selective_build PRIVATE executorch_core program_schema)

# Install the module
install(TARGETS selective_build LIBRARY DESTINATION executorch/codegen/tools)
