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

set(_test_srcs test_diffusion_runner.cpp test_metadata.cpp test_pipelines.cpp
               test_schedulers.cpp test_text_inputs.cpp test_torch_rng.cpp
)

set(_test_pte_files
    "${CMAKE_CURRENT_BINARY_DIR}/DiffusionMetadata_flux.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/DiffusionMetadata_sd15.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/DiffusionMetadata_bare.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/tiny_sd15.pte"
    "${CMAKE_CURRENT_BINARY_DIR}/tiny_flux.pte"
)
# The programs are exported with the Diffusers export code and behaviors, so a
# change to any of them exports them again.
file(GLOB_RECURSE _diffusers_python_srcs CONFIGURE_DEPENDS
     "${EXECUTORCH_ROOT}/export/huggingface/diffusers/*.py"
     "${EXECUTORCH_ROOT}/extension/diffusers/*.py"
)
add_custom_command(
  OUTPUT ${_test_pte_files}
  COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/export_test_programs.py
    --outdir ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_tokenizer.json
          ${_diffusers_python_srcs}
)
add_custom_target(generated_diffusion_test_programs DEPENDS ${_test_pte_files})

et_cxx_test(
  test_diffusers_runner
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  executorch
  extension_diffusers_runner
  portable_kernels
  portable_ops_lib
)
add_dependencies(test_diffusers_runner generated_diffusion_test_programs)
set_property(
  TEST test_diffusers_runner
  PROPERTY
    ENVIRONMENT
    "DIFFUSERS_GOLDENS_PATH=${CMAKE_CURRENT_SOURCE_DIR}/goldens.json;DIFFUSERS_PROGRAMS_DIR=${CMAKE_CURRENT_BINARY_DIR};DIFFUSERS_TOKENIZER_PATH=${CMAKE_CURRENT_SOURCE_DIR}/test_tokenizer.json;DIFFUSERS_SENTENCEPIECE_MODEL=${EXECUTORCH_ROOT}/extension/llm/tokenizers/test/resources/test_sentencepiece.model"
)
