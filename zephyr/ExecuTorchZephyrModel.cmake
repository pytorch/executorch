# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${EXECUTORCH_DIR}/tools/cmake/ExportModel.cmake)

# Resolve a model path from user configuration and handle ET_PTE_FILE_PATH
# and/or CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT.
#
# If CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT is not set, return ET_PTE_FILE_PATH.
#
# If a model was converted by a custom Python exporter script, the output file
# is saved/cached in EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT so reruns can detect if
# the user points ET_PTE_FILE_PATH at the same file and avoid regenerating it.
# If auto export is not used EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT will be
# cleared.
#
# If ET_PTE_FILE_PATH is set and not equal to
# EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT or the name of the generated pte, clear
# the EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT cmake cache value and return
# ET_PTE_FILE_PATH.
#
# Otherwise, try finding the specified model by absolute path, in the CMake
# source, or zephyr source.
#
# If a model is auto-generated, save its output path so later configure-time
# logic can recognize it as build-generated output.
function(executorch_zephyr_resolve_model out_var)

  get_filename_component(_et_zephyr_workspace_dir "${ZEPHYR_BASE}/../" ABSOLUTE)
  set(_et_pte_file_path "${ET_PTE_FILE_PATH}")
  set(_et_has_export_script FALSE)
  if(DEFINED CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT)
    if(NOT "${CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT}" STREQUAL "")
      set(_et_has_export_script TRUE)
    endif()
  endif()

  # EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT caches the generated file from the
  # previous configure step. Read it first, then clear it so each configure pass
  # recomputes the active generated output from the current settings.
  set(_et_auto_model_output "${EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT}")

  set(EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT
      ""
      CACHE INTERNAL "Auto-generated ExecuTorch model path" FORCE
  )
  set_property(DIRECTORY PROPERTY EXECUTORCH_ZEPHYR_GENERATED_MODEL_TARGET "")
  set_property(DIRECTORY PROPERTY EXECUTORCH_ZEPHYR_GENERATED_MODEL_OUTPUT "")

  if(_et_has_export_script)
    set(_et_resolved_export_script "${CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT}")
    find_file(
      _et_found_export_script
      NAMES "${_et_resolved_export_script}"
      HINTS "${CMAKE_CURRENT_SOURCE_DIR}" "${_et_zephyr_workspace_dir}"
      PATHS ""
    )
    if(NOT _et_found_export_script OR NOT EXISTS "${_et_found_export_script}")
      message(
        FATAL_ERROR
          "CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT points to a missing file: ${CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT}"
      )
    endif()

    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_et_found_export_script}"
    )

    if(NOT "${CONFIG_EXECUTORCH_EXPORT_PYTHON_DEPENDENCIES}" STREQUAL "")
      separate_arguments(
        _et_export_python_dependencies NATIVE_COMMAND
        "${CONFIG_EXECUTORCH_EXPORT_PYTHON_DEPENDENCIES}"
      )

      foreach(_et_export_python_dependency IN
              LISTS _et_export_python_dependencies
      )
        if(IS_ABSOLUTE "${_et_export_python_dependency}")
          set(_et_found_export_dependency "${_et_export_python_dependency}")
        elseif(EXISTS
               "${CMAKE_CURRENT_SOURCE_DIR}/${_et_export_python_dependency}"
        )
          get_filename_component(
            _et_found_export_dependency
            "${CMAKE_CURRENT_SOURCE_DIR}/${_et_export_python_dependency}"
            ABSOLUTE
          )
        elseif(EXISTS
               "${_et_zephyr_workspace_dir}/${_et_export_python_dependency}"
        )
          get_filename_component(
            _et_found_export_dependency
            "${_et_zephyr_workspace_dir}/${_et_export_python_dependency}"
            ABSOLUTE
          )
        else()
          message(
            FATAL_ERROR
              "CONFIG_EXECUTORCH_EXPORT_PYTHON_DEPENDENCIES points to a missing file: ${_et_export_python_dependency}"
          )
        endif()

        set_property(
          DIRECTORY
          APPEND
          PROPERTY CMAKE_CONFIGURE_DEPENDS "${_et_found_export_dependency}"
        )
      endforeach()
    endif()

    set(_et_export_python_working_directory "${CMAKE_CURRENT_BINARY_DIR}")
    if(NOT "${CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT}" STREQUAL "")
      if(IS_ABSOLUTE "${CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT}")
        set(_et_generated_pte
            "${CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT}"
        )
      else()
        set(_et_generated_pte
            "${_et_export_python_working_directory}/${CONFIG_EXECUTORCH_EXPORT_PYTHON_GENERATED_OUTPUT}"
        )
      endif()
    else()
      set(_et_generated_pte "${CMAKE_CURRENT_BINARY_DIR}/configured_model.pte")
    endif()

    if(_et_pte_file_path
       AND NOT _et_pte_file_path STREQUAL _et_auto_model_output
       AND NOT _et_pte_file_path STREQUAL _et_generated_pte
    )
      message(
        STATUS
          "ET_PTE_FILE_PATH is set to a different file than the generated model, using ET_PTE_FILE_PATH: ${_et_pte_file_path} and ignoring CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT: ${CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT}"
      )
    else()
      set_property(
        DIRECTORY PROPERTY EXECUTORCH_ZEPHYR_GENERATED_MODEL_OUTPUT
                           "${_et_generated_pte}"
      )

      set(_et_export_python_args)
      if(NOT "${CONFIG_EXECUTORCH_EXPORT_PYTHON_ARGS}" STREQUAL "")
        separate_arguments(
          _et_export_python_args NATIVE_COMMAND
          "${CONFIG_EXECUTORCH_EXPORT_PYTHON_ARGS}"
        )
      endif()

      message(
        STATUS
          "Generating model from ${CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT} with Python exporter"
      )
      executorch_run_python_exporter(
        _et_generated_model_path
        SCRIPT
        "${_et_found_export_script}"
        OUTPUT
        "${_et_generated_pte}"
        WORKING_DIRECTORY
        "${_et_export_python_working_directory}"
        PYTHON_EXECUTABLE
        "${Python3_EXECUTABLE}"
        ARGS
        ${_et_export_python_args}
      )

      set(_et_pte_file_path "${_et_generated_model_path}")

      if(NOT EXISTS "${_et_generated_model_path}")
        message(
          FATAL_ERROR
            "Generated model file does not exist: ${_et_generated_model_path}"
        )
      endif()
      set(EXECUTORCH_ZEPHYR_AUTO_MODEL_OUTPUT
          "${_et_generated_model_path}"
          CACHE INTERNAL "Auto-generated ExecuTorch model path" FORCE
      )
    endif()
  endif()

  if(_et_pte_file_path AND NOT IS_ABSOLUTE "${_et_pte_file_path}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_et_pte_file_path}")
      get_filename_component(
        _et_pte_file_path "${CMAKE_CURRENT_SOURCE_DIR}/${_et_pte_file_path}"
        ABSOLUTE
      )
    elseif(EXISTS "${_et_zephyr_workspace_dir}/${_et_pte_file_path}")
      get_filename_component(
        _et_pte_file_path "${_et_zephyr_workspace_dir}/${_et_pte_file_path}"
        ABSOLUTE
      )
    else()
      get_filename_component(
        _et_pte_file_path "${_et_pte_file_path}" ABSOLUTE BASE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}"
      )
    endif()
  endif()

  set(${out_var}
      "${_et_pte_file_path}"
      PARENT_SCOPE
  )
endfunction()

# Convert a model to a .(b)pte file if CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT is
# set, otherwise rely on ET_PTE_FILE_PATH to be set by the user. Also check if
# the model contains any non-delegated ops and configure selective build state.
#
# The first output variable receives the resolved model path. A second output
# variable may be provided to receive whether any non-delegated ops were found.
function(executorch_zephyr_prepare_model_for_build out_path_var)
  cmake_parse_arguments(ARG "REQUIRE_BPTE" "" "" ${ARGN})
  set(out_has_ops_var "${ARGV1}")

  executorch_zephyr_resolve_model(_et_pte_file_path)

  if(NOT _et_pte_file_path)
    message(
      FATAL_ERROR
        "Set ET_PTE_FILE_PATH or CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT to provide an ExecuTorch model to embed."
    )
  endif()

  if(NOT IS_ABSOLUTE "${_et_pte_file_path}")
    message(FATAL_ERROR "ET_PTE_FILE_PATH must resolve to an absolute path")
  endif()

  if(NOT EXISTS "${_et_pte_file_path}")
    message(
      FATAL_ERROR
        "Could not find ExecuTorch model at ET_PTE_FILE_PATH: ${_et_pte_file_path}"
    )
  endif()

  if(ARG_REQUIRE_BPTE AND NOT _et_pte_file_path MATCHES [[\.bpte$]])
    message(
      FATAL_ERROR
        "This sample requires a .bpte model, got: ${_et_pte_file_path}"
    )
  endif()

  set(ET_PTE_FILE_PATH
      "${_et_pte_file_path}"
      CACHE FILEPATH "Path to the ExecuTorch .pte (or .bpte) model to embed"
            FORCE
  )

  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} "${EXECUTORCH_DIR}/codegen/tools/gen_oplist.py"
      --model_file_path=${_et_pte_file_path}
      --output_path=${CMAKE_CURRENT_BINARY_DIR}/temp.yaml
    RESULT_VARIABLE _et_gen_oplist_status
    OUTPUT_VARIABLE _et_gen_oplist_output
    ERROR_VARIABLE _et_gen_oplist_error
  )

  if(NOT _et_gen_oplist_status EQUAL 0)
    message(
      FATAL_ERROR
        "gen_oplist.py failed for model ${_et_pte_file_path}\nstdout:\n${_et_gen_oplist_output}\nstderr:\n${_et_gen_oplist_error}"
    )
  endif()

  if(_et_gen_oplist_output MATCHES "aten::" OR _et_gen_oplist_output MATCHES
                                               "dim_order_ops::"
  )
    set(_et_found_ops_in_file TRUE)
  else()
    set(_et_found_ops_in_file FALSE)
  endif()

  if(_et_found_ops_in_file)
    set(EXECUTORCH_SELECT_OPS_LIST
        ""
        PARENT_SCOPE
    )
    set(EXECUTORCH_SELECT_OPS_MODEL
        "${_et_pte_file_path}"
        CACHE STRING "Select operators from this ExecuTorch model" FORCE
    )
    set(_EXECUTORCH_GEN_ZEPHYR_PORTABLE_OPS
        ON
        PARENT_SCOPE
    )
    message(
      "gen_oplist:  EXECUTORCH_SELECT_OPS_MODEL=${_et_pte_file_path} is used to auto generate ops from"
    )
  else()
    set(EXECUTORCH_SELECT_OPS_LIST
        ""
        PARENT_SCOPE
    )
    set(EXECUTORCH_SELECT_OPS_MODEL
        ""
        CACHE STRING "Select operators from this ExecuTorch model" FORCE
    )
    set(_EXECUTORCH_GEN_ZEPHYR_PORTABLE_OPS
        OFF
        PARENT_SCOPE
    )
    message(
      "gen_oplist: No non delegated ops were found in ${_et_pte_file_path}; no portable ops added to build"
    )
  endif()

  set(${out_path_var}
      "${_et_pte_file_path}"
      PARENT_SCOPE
  )
  if(out_has_ops_var)
    set(${out_has_ops_var}
        "${_et_found_ops_in_file}"
        PARENT_SCOPE
    )
  endif()
endfunction()

# Generate a model .h header from a .pte file and add it to the specified
# target's include path.
function(executorch_zephyr_add_model_header target_name model_path section)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Target '${target_name}' does not exist")
  endif()

  if(NOT IS_ABSOLUTE "${model_path}")
    message(
      FATAL_ERROR
        "Model path passed to executorch_zephyr_add_model_header must be absolute: ${model_path}"
    )
  endif()

  if(NOT EXISTS "${model_path}")
    message(FATAL_ERROR "Model file does not exist: ${model_path}")
  endif()

  set(_model_pte_header "${CMAKE_CURRENT_BINARY_DIR}/model_pte.h")
  get_property(
    _generated_model_target
    DIRECTORY
    PROPERTY EXECUTORCH_ZEPHYR_GENERATED_MODEL_TARGET
  )
  get_property(
    _generated_model_output
    DIRECTORY
    PROPERTY EXECUTORCH_ZEPHYR_GENERATED_MODEL_OUTPUT
  )
  set(_model_header_depends
      ${model_path}
      ${EXECUTORCH_DIR}/examples/arm/executor_runner/pte_to_header.py
  )
  if(_generated_model_target AND _generated_model_output STREQUAL
                                 "${model_path}"
  )
    list(APPEND _model_header_depends ${_generated_model_target})
  endif()
  add_custom_command(
    OUTPUT ${_model_pte_header}
    COMMAND
      ${Python3_EXECUTABLE}
      ${EXECUTORCH_DIR}/examples/arm/executor_runner/pte_to_header.py --pte
      ${model_path} --outdir ${CMAKE_CURRENT_BINARY_DIR} --section ${section}
    DEPENDS ${_model_header_depends}
    COMMENT "Converting ${model_path} to model_pte.h"
  )

  set(_model_header_target gen_model_header_${target_name})
  add_custom_target(${_model_header_target} DEPENDS ${_model_pte_header})
  add_dependencies(${target_name} ${_model_header_target})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()
