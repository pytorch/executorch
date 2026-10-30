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
  runtime/deserialize/test_samples.cpp
  runtime/graph/test_aliases.cpp
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

# Judges an engine's outputs against a package's golden and precise samples.
add_library(native_output_check STATIC samples/OutputCheck.cpp)
target_link_libraries(native_output_check PUBLIC native_package)
add_executable(native_output_check_test samples/test_output_check.cpp)
target_link_libraries(
  native_output_check_test PRIVATE native_output_check GTest::gtest_main
)
add_test(NAME native_output_check_test COMMAND native_output_check_test)
foreach(target native_output_check native_output_check_test)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endforeach()

# Op packages with bundled samples, one per op and quantization scheme. The
# environment pins the eager kernels' ISA and threads, so the golden is the same
# on any x86 host with AVX2. Other hosts can differ.
set(_native_op_samples ${CMAKE_CURRENT_BINARY_DIR}/op_samples)
file(GLOB_RECURSE _native_op_samples_srcs CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/samples/*.py
     ${EXECUTORCH_ROOT}/backends/cadence/utils/FACTO/facto/*.py
)
add_custom_command(
  OUTPUT ${_native_op_samples}.stamp
  COMMAND
    ${CMAKE_COMMAND} -E env MKL_CBWR=COMPATIBLE ONEDNN_MAX_CPU_ISA=AVX2
    ATEN_CPU_CAPABILITY=avx2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "PYTHONPATH=${EXECUTORCH_ROOT}/src:${EXECUTORCH_ROOT}/backends/cadence/utils/FACTO"
    ${PYTHON_EXECUTABLE} -m executorch.backends.native.test.samples.generate
    --out ${_native_op_samples}
  COMMAND ${CMAKE_COMMAND} -E touch ${_native_op_samples}.stamp
  DEPENDS ${_native_op_samples_srcs}
          ${EXECUTORCH_ROOT}/backends/test/suite/operators/test_linear.py
  VERBATIM
)
add_custom_target(native_op_samples DEPENDS ${_native_op_samples}.stamp)

if(EXECUTORCH_BUILD_NATIVE_GGML OR EXECUTORCH_BUILD_NATIVE_VULKAN)
  if(NOT TARGET gflags AND NOT TARGET gflags::gflags)
    find_package(gflags REQUIRED)
  endif()
  if(TARGET gflags::gflags)
    set(_native_test_gflags gflags::gflags)
  else()
    set(_native_test_gflags gflags)
  endif()

  # A gtest main judging the engine whose host a test links on the bundled
  # samples of the packages it names; engine tests set its flags.
  add_library(native_sample_test_main OBJECT samples/sample_test_main.cpp)
  target_link_libraries(
    native_sample_test_main
    PUBLIC native_output_check native_runtime native_package native_engine
           ${_native_test_gflags} GTest::gtest
  )
  target_compile_options(
    native_sample_test_main
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()

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

  add_executable(native_vulkan_sample_test runtime/vulkan/vulkan_ptn_runner.cpp)
  target_link_libraries(
    native_vulkan_sample_test PRIVATE native_sample_test_main
                                      native_vulkan_engine
  )
  add_dependencies(native_vulkan_sample_test native_op_samples)
  add_test(NAME native_vulkan_sample_test
           COMMAND native_vulkan_sample_test --packages=${_native_op_samples}
  )
  target_compile_options(
    native_vulkan_sample_test
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

  add_executable(native_ggml_sample_test runtime/ggml/ggml_ptn_runner.cpp)
  target_link_libraries(
    native_ggml_sample_test PRIVATE native_sample_test_main native_ggml_engine
  )
  add_dependencies(native_ggml_sample_test native_op_samples)
  add_test(NAME native_ggml_sample_test
           COMMAND native_ggml_sample_test --packages=${_native_op_samples}
                   --device=cpu
  )
  # native_ggml_sample_test_act_quant lets ggml quantize the activations of
  # weight-only quantized linears, judged against golden-int8_act.
  add_test(NAME native_ggml_sample_test_act_quant
           COMMAND native_ggml_sample_test --packages=${_native_op_samples}
                   --device=cpu --allow_activation_quant_for_weight_only
  )
  target_compile_options(
    native_ggml_sample_test
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

if(EXECUTORCH_BUILD_NATIVE_CPU)
  add_executable(native_cpu_engine_test runtime/cpu/test_cpu_engine.cpp)
  target_link_libraries(
    native_cpu_engine_test
    PRIVATE native_cpu_engine cpu_reference extension_threadpool native_api
            native_graph_schema GTest::gtest_main
  )
  add_dependencies(native_cpu_engine_test native_graph_schema_gen)
  add_test(NAME native_cpu_engine_test COMMAND native_cpu_engine_test)
  target_compile_options(
    native_cpu_engine_test
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()
