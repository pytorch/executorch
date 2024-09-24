# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains util functions to generate code for kernel registration for
# both AOT and runtime.

# Selective build. See codegen/tools/gen_oplist.py for how to use these
# arguments.
function(gen_selected_ops)
  set(arg_names LIB_NAME OPS_SCHEMA_YAML ROOT_OPS INCLUDE_ALL_OPS)
  cmake_parse_arguments(GEN "" "" "${arg_names}" ${ARGN})

  message(STATUS "Generating operator lib:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  OPS_SCHEMA_YAML: ${GEN_OPS_SCHEMA_YAML}")
  message(STATUS "  ROOT_OPS: ${GEN_ROOT_OPS}")
  message(STATUS "  INCLUDE_ALL_OPS: ${GEN_INCLUDE_ALL_OPS}")

  set(_oplist_yaml
      ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME}/selected_operators.yaml
  )
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})

  file(GLOB_RECURSE _codegen_tools_srcs "${EXECUTORCH_ROOT}/codegen/tools/*.py")

  set(_gen_oplist_command "${PYTHON_EXECUTABLE}" -m codegen.tools.gen_oplist
                          --output_path=${_oplist_yaml}
  )

  if(GEN_OPS_SCHEMA_YAML)
    list(APPEND _gen_oplist_command
         --ops_schema_yaml_path="${GEN_OPS_SCHEMA_YAML}"
    )
  endif()
  if(GEN_ROOT_OPS)
    list(APPEND _gen_oplist_command --root_ops="${GEN_ROOT_OPS}")
  endif()
  if(GEN_INCLUDE_ALL_OPS)
    list(APPEND _gen_oplist_command --include_all_operators)
  endif()

  message("Command - ${_gen_oplist_command}")
  add_custom_command(
    COMMENT "Generating selected_operators.yaml for ${GEN_LIB_NAME}"
    OUTPUT ${_oplist_yaml}
    COMMAND ${_gen_oplist_command}
    DEPENDS ${GEN_OPS_SCHEMA_YAML} ${_codegen_tools_srcs}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  )

endfunction()

# Codegen for registering kernels. Kernels are defined in functions_yaml and
# custom_ops_yaml.
#
# Invoked as generate_bindings_for_kernels( LIB_NAME lib_name FUNCTIONS_YAML
# functions_yaml CUSTOM_OPS_YAML custom_ops_yaml )
function(generate_bindings_for_kernels)
  set(arg_names LIB_NAME FUNCTIONS_YAML CUSTOM_OPS_YAML)
  cmake_parse_arguments(GEN "" "${arg_names}" "" ${ARGN})

  message(STATUS "Generating kernel bindings:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  FUNCTIONS_YAML: ${GEN_FUNCTIONS_YAML}")
  message(STATUS "  CUSTOM_OPS_YAML: ${GEN_CUSTOM_OPS_YAML}")

  # Command to generate selected_operators.yaml from custom_ops.yaml.
  file(GLOB_RECURSE _codegen_templates "${EXECUTORCH_ROOT}/codegen/templates/*")

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})
  # By default selective build output is selected_operators.yaml
  set(_oplist_yaml ${_out_dir}/selected_operators.yaml)

  # Command to codegen C++ wrappers to register custom ops to both PyTorch and
  # Executorch runtime.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "from distutils.sysconfig import get_python_lib;print(get_python_lib())"
    OUTPUT_VARIABLE site-packages-out
    ERROR_VARIABLE site-packages-out-error
    RESULT_VARIABLE site-packages-result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(GLOB_RECURSE _torchgen_srcs "${site-packages-out}/torchgen/*.py")
  set(_gen_command
      "${PYTHON_EXECUTABLE}" -m torchgen.gen_executorch
      --source-path=${EXECUTORCH_ROOT}/codegen --install-dir=${_out_dir}
      --tags-path=${site-packages-out}/torchgen/packaged/ATen/native/tags.yaml
      --aten-yaml-path=${site-packages-out}/torchgen/packaged/ATen/native/native_functions.yaml
      --op-selection-yaml-path=${_oplist_yaml}
  )

  set(_gen_command_sources
      ${_out_dir}/RegisterCodegenUnboxedKernelsEverything.cpp
      ${_out_dir}/Functions.h ${_out_dir}/NativeFunctions.h
  )

  if(GEN_FUNCTIONS_YAML)
    list(APPEND _gen_command --functions-yaml-path=${GEN_FUNCTIONS_YAML})
  endif()
  if(GEN_CUSTOM_OPS_YAML)
    list(APPEND _gen_command --custom-ops-yaml-path=${GEN_CUSTOM_OPS_YAML})
    list(APPEND _gen_command_sources ${_out_dir}/RegisterCPUCustomOps.cpp
         ${_out_dir}/RegisterSchema.cpp ${_out_dir}/CustomOpsNativeFunctions.h
    )
  endif()

  add_custom_command(
    COMMENT "Generating code for kernel registration"
    OUTPUT ${_gen_command_sources}
    COMMAND ${_gen_command}
    DEPENDS ${_oplist_yaml} ${GEN_CUSTOM_OPS_YAML} ${GEN_FUNCTIONS_YAML}
            ${_codegen_templates} ${_torchgen_srcs}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  )
  # Make generated file list available in parent scope
  set(gen_command_sources
      ${_gen_command_sources}
      PARENT_SCOPE
  )
endfunction()

# Generate an AOT lib for registering custom ops into PyTorch
function(gen_custom_ops_aot_lib)
  cmake_parse_arguments(GEN "" "LIB_NAME" "KERNEL_SOURCES" ${ARGN})
  message(STATUS "Generating custom ops aot lib:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  foreach(SOURCE IN LISTS GEN_KERNEL_SOURCES)
    message(STATUS "  KERNEL_SOURCE: ${SOURCE}")
  endforeach()

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})
  add_library(
    ${GEN_LIB_NAME} SHARED
    ${_out_dir}/RegisterCPUCustomOps.cpp ${_out_dir}/RegisterSchema.cpp
    ${_out_dir}/CustomOpsNativeFunctions.h "${GEN_KERNEL_SOURCES}"
  )
  # Find `Torch`.
  find_package(Torch REQUIRED)
  # This lib uses ATen lib, so we explicitly enable rtti and exceptions.
  target_compile_options(${GEN_LIB_NAME} PRIVATE -frtti -fexceptions)
  target_compile_definitions(${GEN_LIB_NAME} PRIVATE USE_ATEN_LIB=1)
  include_directories(${TORCH_INCLUDE_DIRS})
  target_link_libraries(${GEN_LIB_NAME} PRIVATE torch)

  include(${EXECUTORCH_ROOT}/build/Utils.cmake)

  target_link_options_shared_lib(${GEN_LIB_NAME})
  if(TARGET portable_lib)
    target_link_libraries(${GEN_LIB_NAME} PRIVATE portable_lib)
  else()
    target_link_libraries(${GEN_LIB_NAME} PRIVATE executorch_no_prim_ops)
  endif()
endfunction()

# Generate a runtime lib for registering operators in Executorch
function(gen_operators_lib)
  set(multi_arg_names LIB_NAME KERNEL_LIBS DEPS)
  cmake_parse_arguments(GEN "" "" "${multi_arg_names}" ${ARGN})

  message(STATUS "Generating operator lib:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  KERNEL_LIBS: ${GEN_KERNEL_LIBS}")
  message(STATUS "  DEPS: ${GEN_DEPS}")

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})

  add_library(${GEN_LIB_NAME})
  target_sources(
    ${GEN_LIB_NAME}
    PRIVATE ${_out_dir}/RegisterCodegenUnboxedKernelsEverything.cpp
            ${_out_dir}/Functions.h ${_out_dir}/NativeFunctions.h
  )
  target_link_libraries(${GEN_LIB_NAME} PRIVATE ${GEN_DEPS})
  if(GEN_KERNEL_LIBS)
    target_link_libraries(${GEN_LIB_NAME} PUBLIC ${GEN_KERNEL_LIBS})
  endif()

  target_link_options_shared_lib(${GEN_LIB_NAME})
  set(_generated_headers ${_out_dir}/Functions.h ${_out_dir}/NativeFunctions.h)
  set_target_properties(
    ${GEN_LIB_NAME} PROPERTIES PUBLIC_HEADER "${_generated_headers}"
  )
endfunction()

# Merge two kernel yaml files, prioritizing functions from FUNCTIONS_YAML and
# taking functions from FALLBACK_YAML when no implementation is found. This
# corresponds to the merge_yaml buck implementation in codegen/tools.
function(merge_yaml)
  set(arg_names FUNCTIONS_YAML FALLBACK_YAML OUTPUT_DIR)
  cmake_parse_arguments(GEN "" "${arg_names}" "" ${ARGN})
  message(STATUS "Merging kernel yaml files:")
  message(STATUS "  FUNCTIONS_YAML: ${GEN_FUNCTIONS_YAML}")
  message(STATUS "  FALLBACK_YAML: ${GEN_FALLBACK_YAML}")
  message(STATUS "  OUTPUT_DIR: ${GEN_OUTPUT_DIR}")

  set(_gen_command
      "${PYTHON_EXECUTABLE}" -m codegen.tools.merge_yaml
      --functions_yaml_path=${GEN_FUNCTIONS_YAML}
      --fallback_yaml_path=${GEN_FALLBACK_YAML} --output_dir=${GEN_OUTPUT_DIR}
  )

  add_custom_command(
    COMMENT "Merging kernel yaml files"
    OUTPUT ${GEN_OUTPUT_DIR}/merged.yaml
    COMMAND ${_gen_command}
    DEPENDS ${GEN_FUNCTIONS_YAML} ${GEN_FALLBACK_YAML}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  )
endfunction()
