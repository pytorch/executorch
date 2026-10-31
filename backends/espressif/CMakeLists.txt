# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

idf_component_register(
  SRCS "ops/op_quantized_mul.cpp" PRIV_REQUIRES esp-nn WHOLE_ARCHIVE
)

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)
idf_build_get_property(PYTHON_EXECUTABLE PYTHON)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

set(_yaml_file "${CMAKE_CURRENT_SOURCE_DIR}/ops/operators.yaml")
gen_selected_ops(LIB_NAME espressif_ops_lib OPS_SCHEMA_YAML "${_yaml_file}")
generate_bindings_for_kernels(
  LIB_NAME espressif_ops_lib CUSTOM_OPS_YAML "${_yaml_file}"
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
target_link_libraries(${COMPONENT_LIB} PUBLIC executorch_core)
