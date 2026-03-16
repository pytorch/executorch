# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains util functions to generate code for kernel registration for
# both AOT and runtime.

# Selective build. See codegen/tools/gen_oplist.py for how to use these
# arguments.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

function(gen_selected_ops)
  set(arg_names LIB_NAME OPS_SCHEMA_YAML ROOT_OPS INCLUDE_ALL_OPS
                OPS_FROM_MODEL DTYPE_SELECTIVE_BUILD
  )
  cmake_parse_arguments(GEN "" "" "${arg_names}" ${ARGN})

  message(STATUS "Generating selected operator lib:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  OPS_SCHEMA_YAML: ${GEN_OPS_SCHEMA_YAML}")
  message(STATUS "  ROOT_OPS: ${GEN_ROOT_OPS}")
  message(STATUS "  INCLUDE_ALL_OPS: ${GEN_INCLUDE_ALL_OPS}")
  message(STATUS "  OPS_FROM_MODEL: ${GEN_OPS_FROM_MODEL}")
  message(STATUS "  DTYPE_SELECTIVE_BUILD: ${GEN_DTYPE_SELECTIVE_BUILD}")

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})

  if(GEN_DTYPE_SELECTIVE_BUILD)
    if(NOT GEN_OPS_FROM_MODEL)
      message(
        FATAL_ERROR
          "  DTYPE_SELECTIVE_BUILD is only support with model API, please pass in a model"
      )
    endif()
  endif()

  set(_oplist_yaml ${_out_dir}/selected_operators.yaml)

  file(MAKE_DIRECTORY ${_out_dir})

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
  if(GEN_OPS_FROM_MODEL)
    list(APPEND _gen_oplist_command --model_file_path="${GEN_OPS_FROM_MODEL}")
  endif()

  message("Command - ${_gen_oplist_command}")
  add_custom_command(
    COMMENT "Generating selected_operators.yaml for ${GEN_LIB_NAME}"
    OUTPUT ${_oplist_yaml}
    COMMAND ${_gen_oplist_command}
    DEPENDS ${GEN_OPS_SCHEMA_YAML} ${_codegen_tools_srcs}
    WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  )

  if(GEN_DTYPE_SELECTIVE_BUILD)
    set(_opvariant_h ${_out_dir}/selected_op_variants.h)
    set(_gen_opvariant_command
        "${PYTHON_EXECUTABLE}" -m codegen.tools.gen_selected_op_variants
        --yaml-file=${_oplist_yaml} --output-dir=${_out_dir}/
    )
    message("Command - ${_gen_opvariant_command}")
    add_custom_command(
      COMMENT "Generating ${_opvariant_h} for ${GEN_LIB_NAME}"
      OUTPUT ${_opvariant_h}
      COMMAND ${_gen_opvariant_command}
      DEPENDS ${_oplist_yaml} ${GEN_OPS_SCHEMA_YAML} ${_codegen_tools_srcs}
      WORKING_DIRECTORY ${EXECUTORCH_ROOT}
    )
  endif()
endfunction()

# Codegen for registering kernels. Kernels are defined in functions_yaml and
# custom_ops_yaml.
#
# Invoked as generate_bindings_for_kernels( LIB_NAME lib_name FUNCTIONS_YAML
# functions_yaml CUSTOM_OPS_YAML custom_ops_yaml )
function(generate_bindings_for_kernels)
  set(options ADD_EXCEPTION_BOUNDARY)
  set(arg_names LIB_NAME FUNCTIONS_YAML CUSTOM_OPS_YAML DTYPE_SELECTIVE_BUILD)
  cmake_parse_arguments(GEN "${options}" "${arg_names}" "" ${ARGN})

  message(STATUS "Generating kernel bindings:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  FUNCTIONS_YAML: ${GEN_FUNCTIONS_YAML}")
  message(STATUS "  CUSTOM_OPS_YAML: ${GEN_CUSTOM_OPS_YAML}")
  message(STATUS "  ADD_EXCEPTION_BOUNDARY: ${GEN_ADD_EXCEPTION_BOUNDARY}")
  message(STATUS "  DTYPE_SELECTIVE_BUILD: ${GEN_DTYPE_SELECTIVE_BUILD}")

  # Command to generate selected_operators.yaml from custom_ops.yaml.
  file(GLOB_RECURSE _codegen_templates "${EXECUTORCH_ROOT}/codegen/templates/*")

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})
  # By default selective build output is selected_operators.yaml
  set(_oplist_yaml ${_out_dir}/selected_operators.yaml)

  # If dtype selective build is enable, force header file to be preserved
  if(GEN_DTYPE_SELECTIVE_BUILD)
    set(_opvariant_h ${_out_dir}/selected_op_variants.h)
  else()
    set(_opvariant_h "")
  endif()

  # Command to codegen C++ wrappers to register custom ops to both PyTorch and
  # Executorch runtime.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import torchgen;import os; print(os.path.dirname(torchgen.__file__))"
    OUTPUT_VARIABLE torchgen-out
    ERROR_VARIABLE torchgen-out-error
    RESULT_VARIABLE torchgen-result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(GLOB_RECURSE _torchgen_srcs "${torchgen-out}/*.py")
  # Not using module executorch.codegen.gen because it's not installed yet.
  set(_gen_command
      "${PYTHON_EXECUTABLE}" -m codegen.gen
      --source-path=${EXECUTORCH_ROOT}/codegen --install-dir=${_out_dir}
      --tags-path=${torchgen-out}/packaged/ATen/native/tags.yaml
      --aten-yaml-path=${torchgen-out}/packaged/ATen/native/native_functions.yaml
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
    DEPENDS ${_oplist_yaml} ${_opvariant_h} ${GEN_CUSTOM_OPS_YAML}
            ${GEN_FUNCTIONS_YAML} ${_codegen_templates} ${_torchgen_srcs}
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

  executorch_target_link_options_shared_lib(${GEN_LIB_NAME})
  if(TARGET portable_lib)
    target_link_libraries(${GEN_LIB_NAME} PRIVATE portable_lib)
  else()
    target_link_libraries(${GEN_LIB_NAME} PRIVATE executorch_core)
  endif()
endfunction()

# Generate a runtime lib for registering operators in Executorch
function(gen_operators_lib)
  set(multi_arg_names LIB_NAME KERNEL_LIBS DEPS DTYPE_SELECTIVE_BUILD)
  cmake_parse_arguments(GEN "" "" "${multi_arg_names}" ${ARGN})

  message(STATUS "Generating operator lib:")
  message(STATUS "  LIB_NAME: ${GEN_LIB_NAME}")
  message(STATUS "  KERNEL_LIBS: ${GEN_KERNEL_LIBS}")
  message(STATUS "  DEPS: ${GEN_DEPS}")
  message(STATUS "  DTYPE_SELECTIVE_BUILD: ${GEN_DTYPE_SELECTIVE_BUILD}")

  set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME})
  if(GEN_DTYPE_SELECTIVE_BUILD)
    set(_opvariant_h ${_out_dir}/selected_op_variants.h)
  endif()

  add_library(${GEN_LIB_NAME})

  set(_srcs_list ${_out_dir}/RegisterCodegenUnboxedKernelsEverything.cpp
                 ${_out_dir}/Functions.h ${_out_dir}/NativeFunctions.h
  )
  if(GEN_DTYPE_SELECTIVE_BUILD)
    list(APPEND _srcs_list ${_opvariant_h})
  endif()
  target_sources(${GEN_LIB_NAME} PRIVATE ${_srcs_list})
  target_link_libraries(${GEN_LIB_NAME} PRIVATE ${GEN_DEPS})
  set(portable_kernels_check "portable_kernels")
  if(GEN_KERNEL_LIBS)

    set(_common_compile_options -Wno-deprecated-declarations
                                -ffunction-sections -fdata-sections -Os
    )

    if(GEN_DTYPE_SELECTIVE_BUILD)
      if("${portable_kernels_check}" IN_LIST GEN_KERNEL_LIBS)
        list(REMOVE_ITEM GEN_KERNEL_LIBS ${portable_kernels_check})

        # Build kernels_util_all_deps, since later selected_portable_kernels
        # depends on it
        list(TRANSFORM _kernels_util_all_deps__srcs
             PREPEND "${EXECUTORCH_ROOT}/"
        )
        add_library(
          selected_kernels_util_all_deps ${_kernels_util_all_deps__srcs}
        )
        target_link_libraries(
          selected_kernels_util_all_deps PRIVATE executorch_core
        )
        target_include_directories(
          selected_kernels_util_all_deps PUBLIC ${_common_include_directories}
        )
        target_compile_definitions(
          selected_kernels_util_all_deps
          PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
        )
        target_compile_options(
          selected_kernels_util_all_deps PUBLIC ${_common_compile_options}
        )

        # Build selected_portable_kernels
        list(TRANSFORM _portable_kernels__srcs PREPEND "${EXECUTORCH_ROOT}/")
        add_library(selected_portable_kernels ${_portable_kernels__srcs})
        target_link_libraries(
          selected_portable_kernels PRIVATE executorch_core
                                            selected_kernels_util_all_deps
        )
        target_compile_options(
          selected_portable_kernels PUBLIC ${_common_compile_options}
        )
        target_include_directories(
          selected_portable_kernels
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${GEN_LIB_NAME}/
        )

        # Make sure the header is generated before compiling the library
        add_dependencies(selected_portable_kernels ${GEN_LIB_NAME})
        # Create a custom target for the header to ensure proper dependency
        # tracking
        add_custom_target(
          selected_portable_kernels_header DEPENDS ${_opvariant_h}
        )
        add_dependencies(
          selected_portable_kernels selected_portable_kernels_header
        )
        # Apply the compile definition for dtype selective build
        target_compile_definitions(
          selected_portable_kernels PRIVATE EXECUTORCH_SELECTIVE_BUILD_DTYPE=1
        )

        target_link_libraries(${GEN_LIB_NAME} PUBLIC selected_portable_kernels)
      else()
        message(
          FATAL_ERROR
            "Currently dtype selective build is only supported for portable_kernels but {${GEN_KERNEL_LIBS}} were provided!"
        )
      endif()
    endif()

    # After removing portable_kernels, test if there are other kernel libs
    # provided
    if(GEN_KERNEL_LIBS)
      target_link_libraries(${GEN_LIB_NAME} PUBLIC ${GEN_KERNEL_LIBS})
    endif()
  endif()

  executorch_target_link_options_shared_lib(${GEN_LIB_NAME})
  set(_generated_headers ${_out_dir}/Functions.h ${_out_dir}/NativeFunctions.h)
  if(GEN_DTYPE_SELECTIVE_BUILD)
    list(APPEND _generated_headers ${_opvariant_h})
  endif()
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

  # Mark the file as generated to allow it to be referenced from other
  # CMakeLists in the project.
  set_source_files_properties(
    ${GEN_OUTPUT_DIR}/merged.yaml PROPERTIES GENERATED TRUE
  )
endfunction()

# Append the file list in the variable named `name` in build/build_variables.bzl
# to the variable named `outputvar` in the caller's scope.
function(executorch_append_filelist name outputvar)
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
    ${EXECUTORCH_ROOT}/shim_et/xplat/executorch/build/build_variables.bzl
    ${PROJECT_BINARY_DIR}/build_variables.bzl COPYONLY
  )
  if(NOT PYTHON_EXECUTABLE)
    resolve_python_executable()
  endif()
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "exec(open('${EXECUTORCH_ROOT}/shim_et/xplat/executorch/build/build_variables.bzl').read());print(';'.join(${name}))"
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

function(executorch_load_build_variables)
  set(EXECUTORCH_BUILD_VARIABLES_FILELISTS
      EXECUTORCH_SRCS
      EXECUTORCH_CORE_SRCS
      PORTABLE_KERNELS_SRCS
      KERNELS_UTIL_ALL_DEPS_SRCS
      OPTIMIZED_KERNELS_SRCS
      QUANTIZED_KERNELS_SRCS
      OPTIMIZED_CPUBLAS_SRCS
      OPTIMIZED_NATIVE_CPU_OPS_SRCS
      TEST_BACKEND_COMPILER_LIB_SRCS
      EXTENSION_DATA_LOADER_SRCS
      EXTENSION_EVALUE_UTIL_SRCS
      EXTENSION_FLAT_TENSOR_SRCS
      EXTENSION_MEMORY_ALLOCATOR_SRCS
      EXTENSION_MODULE_SRCS
      EXTENSION_NAMED_DATA_MAP_SRCS
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
  set(EXECUTORCH_BUILD_VARIABLES_VARNAMES
      _executorch__srcs
      _executorch_core__srcs
      _portable_kernels__srcs
      _kernels_util_all_deps__srcs
      _optimized_kernels__srcs
      _quantized_kernels__srcs
      _optimized_cpublas__srcs
      _optimized_native_cpu_ops__srcs
      _test_backend_compiler_lib__srcs
      _extension_data_loader__srcs
      _extension_evalue_util__srcs
      _extension_flat_tensor__srcs
      _extension_memory_allocator__srcs
      _extension_module__srcs
      _extension_named_data_map__srcs
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
  foreach(filelist_and_varname IN
          ZIP_LISTS EXECUTORCH_BUILD_VARIABLES_FILELISTS
          EXECUTORCH_BUILD_VARIABLES_VARNAMES
  )
    executorch_append_filelist(
      ${filelist_and_varname_0} "${filelist_and_varname_1}"
    )
    set(${filelist_and_varname_1}
        "${${filelist_and_varname_1}}"
        PARENT_SCOPE
    )
  endforeach()
endfunction()
