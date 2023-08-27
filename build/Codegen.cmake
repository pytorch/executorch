# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains util functions to generate code for kernel registration for
# both AOT and runtime.

# Selective build. See codegen/tools/gen_oplist.py for how to use these
# arguments.
function(gen_selected_ops ops_schema_yaml root_ops include_all_ops)
  set(_oplist_yaml ${CMAKE_CURRENT_BINARY_DIR}/selected_operators.yaml)
  file(GLOB_RECURSE _codegen_tools_srcs "${EXECUTORCH_ROOT}/codegen/tools/*.py")

  set(_gen_oplist_command "${PYTHON_EXECUTABLE}" -m codegen.tools.gen_oplist
                          --output_path=${_oplist_yaml})

  if(ops_schema_yaml)
    list(APPEND _gen_oplist_command --ops_schema_yaml_path="${ops_schema_yaml}")
  endif()
  if(root_ops)
    list(APPEND _gen_oplist_command --root_ops="${root_ops}")
  endif()
  if(include_all_ops)
    list(APPEND _gen_oplist_command
         --include_all_operators="${include_all_ops}")
  endif()

  add_custom_command(
    COMMENT "Generating selected_operators.yaml for custom ops"
    OUTPUT ${_oplist_yaml}
    COMMAND ${_gen_oplist_command}
    DEPENDS ${ops_schema_yaml} ${_codegen_tools_srcs}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT})

endfunction()

# Codegen for registering kernels. Kernels are defined in functions_yaml and
# custom_ops_yaml
function(generate_bindings_for_kernels functions_yaml custom_ops_yaml)
  # Command to generate selected_operators.yaml from custom_ops.yaml.
  file(GLOB_RECURSE _codegen_templates "${EXECUTORCH_ROOT}/codegen/templates/*")
  file(GLOB_RECURSE _torchgen_srcs "${TORCH_ROOT}/torchgen/*.py")
  # By default selective build output is selected_operators.yaml
  set(_oplist_yaml ${CMAKE_CURRENT_BINARY_DIR}/selected_operators.yaml)

  # Command to codegen C++ wrappers to register custom ops to both PyTorch and
  # Executorch runtime.
  set(_gen_command
      "${PYTHON_EXECUTABLE}" -m torchgen.gen_executorch
      --source-path=${EXECUTORCH_ROOT}/codegen
      --install-dir=${CMAKE_CURRENT_BINARY_DIR}
      --tags-path=${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
      --aten-yaml-path=${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
      --op-selection-yaml-path=${_oplist_yaml})

  set(_gen_command_sources
      ${CMAKE_CURRENT_BINARY_DIR}/RegisterCodegenUnboxedKernelsEverything.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/Functions.h
      ${CMAKE_CURRENT_BINARY_DIR}/NativeFunctions.h)

  if(functions_yaml)
    list(APPEND _gen_command --functions-yaml-path=${functions_yaml})
  endif()
  if(custom_ops_yaml)
    list(APPEND _gen_command --custom-ops-yaml-path=${custom_ops_yaml})
    list(
      APPEND
      _gen_command_sources
      ${CMAKE_CURRENT_BINARY_DIR}/RegisterCPUCustomOps.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/RegisterSchema.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/CustomOpsNativeFunctions.h)
  endif()

  add_custom_command(
    COMMENT "Generating code for kernel registration"
    OUTPUT ${_gen_command_sources}
    COMMAND ${_gen_command}
    DEPENDS ${_oplist_yaml} ${custom_ops_yaml} ${functions_yaml}
            ${_codegen_templates} ${_torchgen_srcs}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT})
  # Make generated file list available in parent scope
  set(gen_command_sources
      ${_gen_command_sources}
      PARENT_SCOPE)
endfunction()

# Generate an AOT lib for registering custom ops into PyTorch
function(gen_custom_ops_aot_lib lib_name kernel_sources)
  add_library(
    ${lib_name} SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterCPUCustomOps.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterSchema.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/CustomOpsNativeFunctions.h ${kernel_sources})
  # Find `Torch`.
  find_package(Torch REQUIRED)
  target_compile_definitions(${lib_name} PRIVATE USE_ATEN_LIB=1)
  include_directories(${TORCH_INCLUDE_DIRS})
  target_link_libraries(${lib_name} PRIVATE torch executorch)

  include(${EXECUTORCH_ROOT}/build/Utils.cmake)

  target_link_options_shared_lib(${lib_name})
endfunction()

# Generate a runtime lib for registering operators in Executorch
function(gen_operators_lib lib_name kernel_lib deps)
  add_library(${lib_name})
  target_sources(
    ${lib_name}
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}/RegisterCodegenUnboxedKernelsEverything.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/Functions.h
      ${CMAKE_CURRENT_BINARY_DIR}/NativeFunctions.h)
  target_link_libraries(${lib_name} PRIVATE ${deps})
  target_link_libraries(${lib_name} INTERFACE ${kernel_lib})

  target_link_options_shared_lib(${lib_name})
endfunction()
