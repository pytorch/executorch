# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains util functions to generate code for kernel registration for
# both AOT and runtime.

# Selective build. See codegen/tools/gen_oplist.py for how to use these
# arguments.
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

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
  set(options ADD_EXCEPTION_BOUNDARY)
  set(arg_names LIB_NAME FUNCTIONS_YAML CUSTOM_OPS_YAML)
  cmake_parse_arguments(GEN "${options}" "${arg_names}" "" ${ARGN})

  message(STATUS "Generating kernel bindings:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  FUNCTIONS_YAML: ${GEN_FUNCTIONS_YAML}")
  message(STATUS "  CUSTOM_OPS_YAML: ${GEN_CUSTOM_OPS_YAML}")
  message(STATUS "  ADD_EXCEPTION_BOUNDARY: ${GEN_ADD_EXCEPTION_BOUNDARY}")

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
  if(GEN_ADD_EXCEPTION_BOUNDARY)
    set(_gen_command "${_gen_command}" --add-exception-boundary)
  endif()

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
  find_package_torch()
  # This lib uses ATen lib, so we explicitly enable rtti and exceptions.
  target_compile_options(${GEN_LIB_NAME} PRIVATE -frtti -fexceptions)
  target_compile_definitions(${GEN_LIB_NAME} PRIVATE USE_ATEN_LIB=1)
  include_directories(${TORCH_INCLUDE_DIRS})
  target_link_libraries(${GEN_LIB_NAME} PRIVATE torch)

  target_link_options_shared_lib(${GEN_LIB_NAME})
  if(TARGET portable_lib)
    target_link_libraries(${GEN_LIB_NAME} PRIVATE portable_lib)
  else()
    target_link_libraries(${GEN_LIB_NAME} PRIVATE executorch_core)
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

# Append the file list in the variable named `name` in
# build/build_variables.bzl to the variable named `outputvar` in the
# caller's scope.
function(append_filelist name outputvar)
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
    ${PROJECT_SOURCE_DIR}/build/build_variables.bzl
    ${PROJECT_BINARY_DIR}/build_variables.bzl COPYONLY
  )
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "exec(open('${PROJECT_SOURCE_DIR}/build/build_variables.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${_rootdir}"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE _tempvar
    ERROR_VARIABLE _stderr
  )
  if(NOT _retval EQUAL 0)
    message(
      FATAL_ERROR
        "Failed to fetch filelist ${name} from build_variables.bzl with output ${_tempvar} and stderr ${_stderr}"
    )
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  list(APPEND ${outputvar} ${_tempvar})
  set(${outputvar}
      "${${outputvar}}"
      PARENT_SCOPE
  )
endfunction()

# Fail the build if the src lists in build_variables.bzl do not match
# the src lists extracted from Buck and placed into
# EXECUTORCH_SRCS_FILE. This is intended to be a safety mechanism
# while we are in the process of removing Buck from the CMake build
# and replacing it with build_variables.bzl; if you are seeing
# failures after you have intentionally changed Buck srcs, then simply
# update build_variables.bzl. If you are seeing failures after
# changing something about the build system, make sure your changes
# will work both before and after we finish replacing Buck with
# build_variables.bzl, which should involve getting these lists to
# match!
function(validate_build_variables)
  include(${EXECUTORCH_SRCS_FILE})
  set(BUILD_VARIABLES_FILELISTS
      EXECUTORCH_SRCS
      EXECUTORCH_CORE_SRCS
      PORTABLE_KERNELS_SRCS
      OPTIMIZED_KERNELS_SRCS
      QUANTIZED_KERNELS_SRCS
      PROGRAM_SCHEMA_SRCS
      OPTIMIZED_CPUBLAS_SRCS
      OPTIMIZED_NATIVE_CPU_OPS_OSS_SRCS
      EXTENSION_DATA_LOADER_SRCS
      EXTENSION_MODULE_SRCS
      EXTENSION_RUNNER_UTIL_SRCS
      EXTENSION_LLM_RUNNER_SRCS
      EXTENSION_TENSOR_SRCS
      EXTENSION_THREADPOOL_SRCS
      EXTENSION_TRAINING_SRCS
      TRAIN_XOR_SRCS
      EXECUTOR_RUNNER_SRCS
      SIZE_TEST_SRCS
      MPS_EXECUTOR_RUNNER_SRCS
      MPS_BACKEND_SRCS
      MPS_SCHEMA_SRCS
      XNN_EXECUTOR_RUNNER_SRCS
      XNNPACK_BACKEND_SRCS
      XNNPACK_SCHEMA_SRCS
      VULKAN_SCHEMA_SRCS
      CUSTOM_OPS_SRCS
      LLAMA_RUNNER_SRCS
  )
  set(BUILD_VARIABLES_VARNAMES
      _executorch__srcs
      _executorch_core__srcs
      _portable_kernels__srcs
      _optimized_kernels__srcs
      _quantized_kernels__srcs
      _program_schema__srcs
      _optimized_cpublas__srcs
      _optimized_native_cpu_ops_oss__srcs
      _extension_data_loader__srcs
      _extension_module__srcs
      _extension_runner_util__srcs
      _extension_llm_runner__srcs
      _extension_tensor__srcs
      _extension_threadpool__srcs
      _extension_training__srcs
      _train_xor__srcs
      _executor_runner__srcs
      _size_test__srcs
      _mps_executor_runner__srcs
      _mps_backend__srcs
      _mps_schema__srcs
      _xnn_executor_runner__srcs
      _xnnpack_backend__srcs
      _xnnpack_schema__srcs
      _vulkan_schema__srcs
      _custom_ops__srcs
      _llama_runner__srcs
  )
  foreach(filelist_and_varname IN ZIP_LISTS BUILD_VARIABLES_FILELISTS
                                  BUILD_VARIABLES_VARNAMES
  )
    append_filelist(
      ${filelist_and_varname_0}
      "${filelist_and_varname_1}_from_build_variables"
    )
    if(NOT ${filelist_and_varname_1} STREQUAL
       ${filelist_and_varname_1}_from_build_variables
    )
      message(
        FATAL_ERROR
          "Buck-generated ${filelist_and_varname_1} does not match hardcoded "
          "${filelist_and_varname_0} in build_variables.bzl. Left: "
          "${${filelist_and_varname_1}}\n "
          "Right: ${${filelist_and_varname_1}_from_build_variables}"
      )
    endif()
  endforeach()
endfunction()
