# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs
    test_generation_config.cpp
    test_hf_static_io_manager.cpp
    test_io_manager.cpp
    test_model_metadata.cpp
    test_text_llm_runner.cpp
    test_text_prefiller.cpp
    test_text_decoder_runner.cpp
    test_multimodal_input.cpp
    test_text_stream.cpp
    test_util.cpp
    test_wav_loader.cpp
)

# Add LSan stub for Apple platforms
if(APPLE)
  list(APPEND _test_srcs lsan_stub.cpp)
endif()

set(_metadata_pte_files
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_full.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_last.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_selected.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_context.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_prefill.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_vocab.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_missing.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_wrong_type.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_mismatched.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_heads.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_dims.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_windows.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_empty.pte"
)
# Local builds may share a host with other model exports.
set(_export_prefix)
if(EXECUTORCH_TEST_EXPORT_LOCK)
  find_program(_flock flock REQUIRED)
  set(_export_prefix ${_flock} ${EXECUTORCH_TEST_EXPORT_LOCK})
endif()

add_custom_command(
  OUTPUT ${_metadata_pte_files}
  COMMAND
    ${_export_prefix} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/export_model_metadata.py
    --outdir ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/export_model_metadata.py
          ${EXECUTORCH_ROOT}/extension/llm/export/model_metadata.py
)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/hf_static_fixtures.stamp"
  COMMAND
    ${_export_prefix} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/export_hf_static_program.py --outdir
    ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/export_hf_static_program.py
)
add_custom_target(
  generated_model_metadata_test_files
  DEPENDS ${_metadata_pte_files}
          "${CMAKE_CURRENT_BINARY_DIR}/hf_static_fixtures.stamp"
)

et_cxx_test(
  test_runner
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  executorch
  extension_llm_runner
  portable_kernels
  portable_ops_lib
)
add_dependencies(test_runner generated_model_metadata_test_files)
set_property(
  TEST test_runner
  PROPERTY
    ENVIRONMENT
    "ET_MODEL_METADATA_FULL_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_full.pte;ET_MODEL_METADATA_LAST_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_last.pte;ET_MODEL_METADATA_SELECTED_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_selected.pte;ET_MODEL_METADATA_INVALID_CONTEXT_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_context.pte;ET_MODEL_METADATA_INVALID_PREFILL_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_prefill.pte;ET_MODEL_METADATA_INVALID_VOCAB_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_invalid_vocab.pte;ET_MODEL_METADATA_MISSING_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_missing.pte;ET_MODEL_METADATA_GEOMETRY_WRONG_TYPE_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_wrong_type.pte;ET_MODEL_METADATA_GEOMETRY_MISMATCHED_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_mismatched.pte;ET_MODEL_METADATA_GEOMETRY_INVALID_HEADS_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_heads.pte;ET_MODEL_METADATA_GEOMETRY_INVALID_DIMS_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_dims.pte;ET_MODEL_METADATA_GEOMETRY_INVALID_WINDOWS_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_invalid_windows.pte;ET_MODEL_METADATA_GEOMETRY_EMPTY_PATH=${CMAKE_CURRENT_BINARY_DIR}/ModelMetadata_geometry_empty.pte;ET_HF_STATIC_TINY_PATH=${CMAKE_CURRENT_BINARY_DIR}/HfStaticTiny.pte;ET_HF_STATIC_TINY_HALF_PATH=${CMAKE_CURRENT_BINARY_DIR}/HfStaticTinyHalf.pte;ET_HF_STATIC_TINY_STATE_PATH=${CMAKE_CURRENT_BINARY_DIR}/HfStaticTinyWithState.pte;ET_HF_STATIC_FIXTURES_DIR=${CMAKE_CURRENT_BINARY_DIR}"
)

# Override sanitizer to this issue:
# https://github.com/abseil/abseil-cpp/issues/841 Root issue:
# https://github.com/llvm/llvm-project/issues/16778
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_options(test_runner PUBLIC --rtlib=compiler-rt)
endif()
