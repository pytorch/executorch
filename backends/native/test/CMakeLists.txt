# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(
  native_runtime_test
  ../runtime/api/test/ApiTypesTest.cpp
  ../runtime/deserialize/test/ReaderBoundsTest.cpp
  ../runtime/deserialize/test/ValidationTest.cpp
  ../runtime/test/MethodMetaTest.cpp
  ../runtime/test/ProgramVersionTest.cpp
  runtime/graph/test_graph.cpp
  runtime/graph/test_graph_utils.cpp
  runtime/graph/test_memory_planning.cpp
  runtime/graph/test_tensor_meta.cpp
)
target_link_libraries(
  native_runtime_test PRIVATE native_api native_graph_schema
                              nlohmann_json::nlohmann_json GTest::gtest_main
)
add_dependencies(native_runtime_test native_graph_schema_gen)
add_test(NAME native_runtime_test COMMAND native_runtime_test)
target_compile_options(
  native_runtime_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                              $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_executable(
    native_vulkan_passes_test
    runtime/vulkan/test_fuse_quantized_embedding.cpp
    runtime/vulkan/test_fuse_quantized_linear.cpp
    runtime/vulkan/test_insert_prepack.cpp
    runtime/vulkan/test_lower_hf_attention.cpp
    runtime/vulkan/test_lower_rms_norm.cpp
    runtime/vulkan/test_materialize_view_copies.cpp
    runtime/vulkan/test_vulkan_constant_materialization_tracker.cpp
    ../runtime/vulkan/VulkanConstantMaterializationTracker.cpp
  )
  target_link_libraries(
    native_vulkan_passes_test PRIVATE native_vulkan_passes GTest::gtest_main
  )
  add_test(NAME native_vulkan_passes_test COMMAND native_vulkan_passes_test)
  target_compile_options(
    native_vulkan_passes_test
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()

if(EXECUTORCH_BUILD_NATIVE_GGML)
  add_executable(native_ggml_engine_test runtime/ggml/test_ggml_engine.cpp)
  target_link_libraries(
    native_ggml_engine_test PRIVATE native_ggml_engine native_graph_schema
                                    GTest::gtest_main
  )
  add_dependencies(native_ggml_engine_test native_graph_schema_gen)
  add_test(NAME native_ggml_engine_test COMMAND native_ggml_engine_test)
  target_compile_options(
    native_ggml_engine_test
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()

if(TARGET native_runner_utils)
  add_executable(
    native_runner_utils_test ../examples/runner_utils/test_runner_utils.cpp
  )
  target_link_libraries(
    native_runner_utils_test PRIVATE native_runner_utils GTest::gtest_main
  )
  add_test(NAME native_runner_utils_test COMMAND native_runner_utils_test)
  target_compile_options(
    native_runner_utils_test
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()

if(TARGET native_llm_runner_utils)
  add_executable(
    native_llm_runner_utils_test
    ../examples/hf_llm/runtime/LlmRunnerUtilsTest.cpp
  )
  target_link_libraries(
    native_llm_runner_utils_test PRIVATE native_llm_runner_utils
                                         GTest::gtest_main
  )
  add_test(NAME native_llm_runner_utils_test
           COMMAND native_llm_runner_utils_test
  )
  target_compile_options(
    native_llm_runner_utils_test
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()
