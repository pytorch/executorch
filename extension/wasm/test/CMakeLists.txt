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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/models/test.pte
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  COMMAND python3 -m extension.wasm.test.test_model
          ${CMAKE_CURRENT_BINARY_DIR}/models/test.pte
)

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
  executorch_wasm_test_models
  DEPENDS ${MODELS_DIR}/add_mul.pte ${MODELS_DIR}/add.pte
          ${MODELS_DIR}/test.pte
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/package.json
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/package.json
          ${CMAKE_CURRENT_BINARY_DIR}/package.json
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package.json
  COMMENT "Copying package.json to build output directory"
)

add_custom_target(
  executorch_wasm_test_package_json
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/package.json
)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  set(ETDUMP_UNIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/unittests_etdump.js)
else()
  set(ETDUMP_UNIT_TESTS
      ${CMAKE_CURRENT_SOURCE_DIR}/unittests_etdump_disabled.js
  )
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unittests_full.js
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/unittests.js ${ETDUMP_UNIT_TESTS} >
          ${CMAKE_CURRENT_BINARY_DIR}/unittests_full.js
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittests.js ${ETDUMP_UNIT_TESTS}
  COMMENT "Copying unittests_full.js to build output directory"
)

add_custom_target(
  executorch_wasm_unittests
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/unittests_full.js
)

add_executable(executorch_wasm_tests)
target_link_libraries(executorch_wasm_tests PRIVATE executorch_wasm)
target_link_options(
  executorch_wasm_tests
  PRIVATE
  --embed-file
  "${MODELS_DIR}@/"
  --pre-js
  ${CMAKE_CURRENT_BINARY_DIR}/unittests_full.js
  -sASSERTIONS=2
)
set_target_properties(
  executorch_wasm_tests PROPERTIES OUTPUT_NAME "executorch_wasm.test"
)
set_property(
  TARGET executorch_wasm_tests
  APPEND
  PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/unittests_full.js
)
add_dependencies(
  executorch_wasm_tests executorch_wasm_unittests executorch_wasm_test_models
  executorch_wasm_test_package_json
)
