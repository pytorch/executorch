# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

add_executable(executorch_wasm_demo_lib)
target_link_libraries(executorch_wasm_demo_lib PUBLIC executorch_wasm)
target_link_options(
  executorch_wasm_demo_lib PUBLIC -sALLOW_MEMORY_GROWTH
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/demo.js
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/demo.js
    ${CMAKE_CURRENT_BINARY_DIR}/demo.js
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/demo.js
  COMMENT "Copying demo.js to build output directory"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/classes.js
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/classes.js
    ${CMAKE_CURRENT_BINARY_DIR}/classes.js
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/classes.js
  COMMENT "Copying classes.js to build output directory"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/demo.html
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/demo.html
          ${CMAKE_CURRENT_BINARY_DIR}/demo.html
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/demo.html
  COMMENT "Copying demo.html to build output directory"
)

add_custom_target(
  executorch_wasm_demo
  DEPENDS executorch_wasm_demo_lib
          ${CMAKE_CURRENT_BINARY_DIR}/classes.js
          ${CMAKE_CURRENT_BINARY_DIR}/demo.js
          ${CMAKE_CURRENT_BINARY_DIR}/demo.html
)
