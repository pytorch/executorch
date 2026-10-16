# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT TARGET tokenizers::tokenizers)
  if(EXISTS "${EXECUTORCH_ROOT}/extension/llm/tokenizers/CMakeLists.txt")
    set(SUPPORT_REGEX_LOOKAHEAD ON)
    add_subdirectory(
      ${EXECUTORCH_ROOT}/extension/llm/tokenizers
      ${CMAKE_CURRENT_BINARY_DIR}/tokenizers
    )
  else()
    find_package(tokenizers CONFIG REQUIRED)
  endif()
endif()

add_library(native_llm_runner_utils STATIC LlmRunnerUtils.cpp)
target_compile_features(native_llm_runner_utils PUBLIC cxx_std_20)
target_include_directories(
  native_llm_runner_utils PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
add_library(
  native_llm_runner STATIC
  LlmRunner.cpp
  ${EXECUTORCH_ROOT}/examples/models/llama/tokenizer/llama_tiktoken.cpp
)
target_link_libraries(
  native_llm_runner
  PUBLIC native_engine
  PRIVATE ${_native_runner_gflags} native_llm_runner_utils native_package
          native_runner_utils native_runtime tokenizers::tokenizers
)
foreach(target native_llm_runner native_llm_runner_utils)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endforeach()

set(_llm_runners)
if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_executable(run_hf_llm_vulkan run_hf_llm_vulkan.cpp)
  target_link_libraries(run_hf_llm_vulkan PRIVATE native_vulkan_engine)
  list(APPEND _llm_runners run_hf_llm_vulkan)
endif()
if(EXECUTORCH_BUILD_NATIVE_GGML)
  add_executable(run_hf_llm_ggml run_hf_llm_ggml.cpp)
  target_link_libraries(run_hf_llm_ggml PRIVATE native_ggml_engine)
  list(APPEND _llm_runners run_hf_llm_ggml)
endif()
foreach(target IN LISTS _llm_runners)
  target_link_libraries(
    ${target} PRIVATE native_llm_runner ${_native_runner_gflags}
  )
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
  install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

if(EXECUTORCH_BUILD_TESTS)
  add_executable(native_llm_runner_utils_test LlmRunnerUtilsTest.cpp)
  target_link_libraries(
    native_llm_runner_utils_test PRIVATE native_llm_runner_utils
                                         GTest::gtest_main
  )
  add_test(NAME native_llm_runner_utils_test
           COMMAND native_llm_runner_utils_test
  )
endif()
