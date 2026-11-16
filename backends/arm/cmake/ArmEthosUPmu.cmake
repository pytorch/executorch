# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(arm_add_ethosu_linux_pmu_events TARGET DRIVER_SOURCE_DIR)
  if(NOT PYTHON_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
  set(_interface_dir "${DRIVER_SOURCE_DIR}/kernel/include/direct/interface")
  set(_u65_header "${_interface_dir}/ethosu_direct_interface_u65.h")
  set(_u85_header "${_interface_dir}/ethosu_direct_interface_u85.h")
  set(_generator
      "${EXECUTORCH_ROOT}/backends/arm/scripts/generate_ethosu_linux_pmu_events.py"
  )
  set(_output_dir "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_generated")
  set(_output "${_output_dir}/EthosULinuxPmuEvents.h")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND "${PYTHON_EXECUTABLE}" "${_generator}" --u65-header "${_u65_header}"
            --u85-header "${_u85_header}" --output "${_output}"
    DEPENDS "${_generator}" "${_u65_header}" "${_u85_header}"
    COMMENT "Generating Linux Ethos-U PMU event translation"
    VERBATIM
  )
  target_sources(${TARGET} PRIVATE "${_output}")
  target_include_directories(${TARGET} PRIVATE "${_output_dir}")
endfunction()
