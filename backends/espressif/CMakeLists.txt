# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

idf_component_register(
  SRCS "ops/op_quantized_mul.cpp" "ops/op_quantized_add.cpp" PRIV_REQUIRES
  esp-nn WHOLE_ARCHIVE
)

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)
idf_build_get_property(PYTHON_EXECUTABLE PYTHON)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT EXECUTORCH_SELECT_OPS_MODEL)
  message(
    FATAL_ERROR
      "Set EXECUTORCH_SELECT_OPS_MODEL to the PTE files used to build the runtime"
  )
endif()

set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/espressif_ops_lib")
set(_esp_yaml "${CMAKE_CURRENT_SOURCE_DIR}/ops/operators.yaml")
set(_quantized_yaml "${EXECUTORCH_ROOT}/kernels/quantized/quantized.yaml")
gen_selected_ops(
  LIB_NAME espressif_ops_lib OPS_FROM_MODEL "${EXECUTORCH_SELECT_OPS_MODEL}"
)
add_custom_command(
  OUTPUT "${_out_dir}/merged.yaml"
  COMMAND
    "${PYTHON_EXECUTABLE}" -m codegen.tools.merge_yaml --functions-yaml-path
    "${_esp_yaml}" --fallback-yaml-path "${_quantized_yaml}" --output_dir
    "${_out_dir}"
  DEPENDS "${_esp_yaml}" "${_quantized_yaml}"
          "${EXECUTORCH_ROOT}/codegen/tools/merge_yaml.py"
  WORKING_DIRECTORY "${EXECUTORCH_ROOT}"
  VERBATIM
)
generate_bindings_for_kernels(
  LIB_NAME espressif_ops_lib FUNCTIONS_YAML
  "${EXECUTORCH_ROOT}/kernels/portable/functions.yaml" CUSTOM_OPS_YAML
  "${_out_dir}/merged.yaml"
)

target_sources(
  ${COMPONENT_LIB}
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/espressif_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp"
)
target_include_directories(
  ${COMPONENT_LIB} PRIVATE "${EXECUTORCH_ROOT}/.."
                           "${EXECUTORCH_ROOT}/runtime/core/portable_type/c10"
)
target_compile_definitions(
  ${COMPONENT_LIB} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)
target_link_libraries(
  ${COMPONENT_LIB} PUBLIC executorch_core portable_kernels quantized_kernels
)
