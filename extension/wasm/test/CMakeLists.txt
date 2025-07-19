# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

set(MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/models/)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/models/add_mul.pte
         ${CMAKE_CURRENT_BINARY_DIR}/models/add.pte
  COMMAND ${CMAKE_COMMAND} -E make_directory "${MODELS_DIR}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  COMMAND python3 -m examples.portable.scripts.export --model_name="add_mul"
          --output_dir="${MODELS_DIR}"
  COMMAND python3 -m examples.portable.scripts.export --model_name="add"
          --output_dir="${MODELS_DIR}"
)

add_custom_target(
  executorch_wasm_test_models DEPENDS ${MODELS_DIR}/add_mul.pte
                                      ${MODELS_DIR}/add.pte
)

add_executable(executorch_wasm_test_lib)
target_link_libraries(executorch_wasm_test_lib PUBLIC executorch_wasm)
target_link_options(
  executorch_wasm_test_lib PUBLIC --embed-file "${MODELS_DIR}@/"
)
add_dependencies(executorch_wasm_test_lib executorch_wasm_test_models)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/executorch_wasm.test.js
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/executorch_wasm.test.js
    ${CMAKE_CURRENT_BINARY_DIR}/executorch_wasm.test.js
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/executorch_wasm.test.js
  COMMENT "Copying executorch_wasm.test.js to build output directory"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/package.json
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/package.json
          ${CMAKE_CURRENT_BINARY_DIR}/package.json
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package.json
  COMMENT "Copying package.json to build output directory"
)

add_custom_target(
  executorch_wasm_tests
  DEPENDS executorch_wasm_test_lib
          ${CMAKE_CURRENT_BINARY_DIR}/executorch_wasm.test.js
          ${CMAKE_CURRENT_BINARY_DIR}/package.json
)
