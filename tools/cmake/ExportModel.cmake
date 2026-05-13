# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_LIST_DIR}/Utils.cmake)

# Shared setup for Python-based model exporters used by both the configure-time
# and build-time helper entry points below.
function(
  _executorch_prepare_python_export
  caller
  script
  output
  working_directory
  python_executable
  out_command_var
  out_output_var
)
  if(NOT script)
    message(FATAL_ERROR "${caller} requires SCRIPT to be set")
  endif()
  if(NOT output)
    message(FATAL_ERROR "${caller} requires OUTPUT to be set")
  endif()
  if(NOT working_directory)
    message(FATAL_ERROR "${caller} requires WORKING_DIRECTORY to be set")
  endif()
  if(NOT IS_DIRECTORY "${working_directory}")
    message(
      FATAL_ERROR
        "${caller} requires WORKING_DIRECTORY to exist: ${working_directory}"
    )
  endif()

  if(NOT python_executable)
    resolve_python_executable()
    set(python_executable ${PYTHON_EXECUTABLE})
  endif()

  if(NOT EXISTS "${script}")
    message(FATAL_ERROR "Python exporter script does not exist: ${script}")
  endif()

  if(IS_ABSOLUTE "${output}")
    set(_et_export_output "${output}")
  else()
    # Resolve relative output paths from the exporter's working directory so
    # configure-time checks and build-time outputs refer to the same file.
    get_filename_component(
      _et_export_output "${output}" ABSOLUTE BASE_DIR "${working_directory}"
    )
  endif()

  get_filename_component(_et_export_output_dir "${_et_export_output}" DIRECTORY)
  if(_et_export_output_dir AND NOT EXISTS "${_et_export_output_dir}")
    file(MAKE_DIRECTORY "${_et_export_output_dir}")
  endif()

  set(_et_export_command ${python_executable} ${script} ${ARGN})
  set(${out_command_var}
      ${_et_export_command}
      PARENT_SCOPE
  )
  set(${out_output_var}
      "${_et_export_output}"
      PARENT_SCOPE
  )
endfunction()

# Run a Python exporter at configure time and require it to produce OUTPUT.
#
# Backend-specific lowering options should be expressed in ARGS and remain owned
# by the Python script. This helper runs the exporter immediately during CMake
# configure, then verifies that the expected output file was written.
function(executorch_run_python_exporter out_var)
  cmake_parse_arguments(
    ARG "" "SCRIPT;OUTPUT;WORKING_DIRECTORY;PYTHON_EXECUTABLE" "ARGS" ${ARGN}
  )

  _executorch_prepare_python_export(
    executorch_run_python_exporter
    "${ARG_SCRIPT}"
    "${ARG_OUTPUT}"
    "${ARG_WORKING_DIRECTORY}"
    "${ARG_PYTHON_EXECUTABLE}"
    _et_export_command
    _et_export_output
    ${ARG_ARGS}
  )

  execute_process(
    COMMAND ${_et_export_command}
    WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
    RESULT_VARIABLE _et_export_status
    OUTPUT_VARIABLE _et_export_stdout
    ERROR_VARIABLE _et_export_stderr
  )

  if(NOT _et_export_status EQUAL 0)
    message(
      FATAL_ERROR
        "Python exporter failed for ${ARG_SCRIPT}\nstdout:\n${_et_export_stdout}\nstderr:\n${_et_export_stderr}"
    )
  endif()

  if(NOT EXISTS "${_et_export_output}")
    message(
      FATAL_ERROR
        "Python exporter ${ARG_SCRIPT} completed but did not produce ${_et_export_output}"
    )
  endif()

  set(${out_var}
      "${_et_export_output}"
      PARENT_SCOPE
  )
endfunction()

# Add a build-time target that runs a Python exporter and materializes OUTPUT.
# Unlike executorch_run_python_exporter(), this helper defers execution until
# the build graph runs and exposes the generated file through a custom target.
function(executorch_add_python_export_target)
  cmake_parse_arguments(
    ARG "" "TARGET;SCRIPT;OUTPUT;WORKING_DIRECTORY;PYTHON_EXECUTABLE"
    "ARGS;DEPENDS" ${ARGN}
  )

  if(NOT ARG_TARGET)
    message(
      FATAL_ERROR
        "executorch_add_python_export_target requires TARGET to be set"
    )
  endif()

  _executorch_prepare_python_export(
    executorch_add_python_export_target
    "${ARG_SCRIPT}"
    "${ARG_OUTPUT}"
    "${ARG_WORKING_DIRECTORY}"
    "${ARG_PYTHON_EXECUTABLE}"
    _et_export_command
    _et_export_output
    ${ARG_ARGS}
  )

  add_custom_command(
    OUTPUT ${_et_export_output}
    COMMAND ${_et_export_command}
    COMMAND_EXPAND_LISTS
    DEPENDS ${ARG_SCRIPT} ${ARG_DEPENDS}
    WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
    COMMENT "Generating model with ${ARG_SCRIPT}"
    VERBATIM
  )

  add_custom_target(${ARG_TARGET} DEPENDS ${_et_export_output})
endfunction()
