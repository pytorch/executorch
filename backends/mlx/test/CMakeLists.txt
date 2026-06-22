# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MLX backend tests

# Strict compiler flags for the test runner — mlxdelegate uses PRIVATE so these
# don't propagate to downstream consumers
set(_mlx_test_compile_options -Wall -Werror -Wconversion -Wsign-conversion
                              -Wshorten-64-to-32
)

# Sanitizers are inherited from parent via EXECUTORCH_MLX_ENABLE_SANITIZERS
if(EXECUTORCH_MLX_ENABLE_SANITIZERS)
  list(APPEND _mlx_test_compile_options -fsanitize=address,undefined
       -fno-omit-frame-pointer
  )
endif()

# Op test runner - generic test binary for testing individual ops
add_executable(op_test_runner op_test_runner.cpp)

target_compile_options(op_test_runner PRIVATE ${_mlx_test_compile_options})
if(EXECUTORCH_MLX_ENABLE_SANITIZERS)
  target_link_options(op_test_runner PRIVATE ${_mlx_sanitizer_link_options})
endif()

target_link_libraries(
  op_test_runner PRIVATE extension_module extension_tensor executorch
                         mlxdelegate
)

# --------------------------------------------------------------------------
# Compile-only strict warnings test for delegate headers
#
# Verifies MLXExecutor.h, MLXInterpreter.h, MLXLoader.h compile cleanly under
# -Wconversion -Wsign-conversion -Wshorten-64-to-32 -Werror. ExecuTorch and MLX
# headers are suppressed via pragma in the source file. This target is never
# linked or run — a successful compile is the test.
# --------------------------------------------------------------------------
add_library(strict_compile_test OBJECT strict_compile_test.cpp)
target_compile_options(strict_compile_test PRIVATE ${_mlx_test_compile_options})
target_include_directories(
  strict_compile_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
)
target_link_libraries(
  strict_compile_test PRIVATE mlx_schema executorch_core mlx
)
add_dependencies(op_test_runner strict_compile_test)

# Multi-threaded inference test
include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

et_cxx_test(
  multi_thread_test_runner
  SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/multi_thread_test_runner.cpp
  EXTRA_LIBS
  extension_module
  extension_tensor
  mlxdelegate
)

# Add sanitizer link flags to multi_thread_test_runner if enabled
if(EXECUTORCH_MLX_ENABLE_SANITIZERS)
  target_link_options(
    multi_thread_test_runner PRIVATE ${_mlx_sanitizer_link_options}
  )
endif()
