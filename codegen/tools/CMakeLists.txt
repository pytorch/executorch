# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check if pybind11 is available

# Create the selective_build pybind11 module
pybind11_add_module(selective_build SHARED selective_build.cpp)
strip_python_lib(selective_build)

# Set the output name to match the module name
set_target_properties(selective_build PROPERTIES OUTPUT_NAME "selective_build")

# Set the module name for the pybind11 module
target_compile_definitions(
  selective_build PUBLIC EXECUTORCH_PYTHON_MODULE_NAME=selective_build
)

# Include directories
target_include_directories(
  selective_build PRIVATE ${EXECUTORCH_SOURCE_INCLUDE_DIR}
)

# Compile options
target_compile_options(
  selective_build
  PUBLIC -Wno-deprecated-declarations
         -fPIC
         -frtti
         -fexceptions
         -Werror
         -Wunused-variable
         -Wno-unknown-argument
)
# We suppress -Wno-unknown-argument because our build system passes -fPIC for
# Unix builds, but we also build on Windows where it's ignored

# Link against required libraries
if(TARGET bundled_program)
  target_compile_definitions(selective_build PRIVATE -DET_BUNDLE_IO)
  target_link_libraries(selective_build PRIVATE bundled_program)
endif()
if(EXECUTORCH_BUILD_SHARED)
  # This module calls into the runtime, so resolve those symbols from
  # libexecutorch.so rather than baking in a second copy of the core. It lands
  # in <site-packages>/executorch/codegen/tools, two levels below the wheel's
  # lib/.
  target_link_libraries(
    selective_build PRIVATE executorch_shared program_schema
  )
  executorch_target_shared_runtime_path(
    selective_build "codegen/tools" "executorch/codegen/tools"
  )
else()
  target_link_libraries(selective_build PRIVATE executorch_core program_schema)
endif()

# Install the module
install(TARGETS selective_build LIBRARY DESTINATION executorch/codegen/tools)
