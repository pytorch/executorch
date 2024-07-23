 # Copyright (c) 2024 MediaTek Inc.
 #
 # Licensed under the BSD License (the "License"); you may not use this file
 # except in compliance with the License. See the license file in the root
 # directory of this source tree for more details.

if(${ANDROID})
    get_filename_component(EXECUTORCH_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../.."
        ABSOLUTE
    )
    set(_mtk_executor_runner__srcs ${_executor_runner__srcs})
    list(
        TRANSFORM
        _mtk_executor_runner__srcs
        PREPEND
        "${EXECUTORCH_SOURCE_DIR}/"
    )
    list(
        FILTER
        _mtk_executor_runner__srcs
        EXCLUDE REGEX
        ".*executor_runner.cpp$"
    )
    list(
        PREPEND
        _mtk_executor_runner__srcs
        ${CMAKE_CURRENT_LIST_DIR}/executor_runner/mtk_executor_runner.cpp
    )

    add_executable(mtk_executor_runner ${_mtk_executor_runner__srcs})

    target_link_libraries(mtk_executor_runner
        ${_executor_runner_libs}
        neuron_backend
        gflags
    )
    target_compile_options(mtk_executor_runner
        PUBLIC
        ${_common_compile_options}
    )

    set(_mtk_llama_executor_runner__srcs ${_mtk_executor_runner__srcs})
    list(
        FILTER
        _mtk_llama_executor_runner__srcs
        EXCLUDE REGEX
        ".*executor_runner.cpp$"
    )
    list(
        PREPEND
        _mtk_llama_executor_runner__srcs
        ${CMAKE_CURRENT_LIST_DIR}/executor_runner/mtk_llama_executor_runner.cpp
    )

    # Build ABSL and RE2
    set(LLAMA2_EXAMPLE_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/models/llama2)
    set(THIRD_PARTY_ABSL_DIR ${LLAMA2_EXAMPLE_MODEL_DIR}/third-party/abseil-cpp)
    set(THIRD_PARTY_RE2_DIR ${LLAMA2_EXAMPLE_MODEL_DIR}/third-party/re2)
    set(ABSL_ENABLE_INSTALL ON)
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_subdirectory(${THIRD_PARTY_ABSL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/third-party/abseil)
    add_subdirectory(${THIRD_PARTY_RE2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/third-party/re2)
    set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

    # Build tokenizers
    set(LLAMA2_TOKENIZER_DIR ${LLAMA2_EXAMPLE_MODEL_DIR}/tokenizer)
    add_library(tokenizer STATIC)
    target_include_directories(tokenizer
        PUBLIC
        ${_common_include_directories}
        ${THIRD_PARTY_ABSL_DIR}
        ${THIRD_PARTY_RE2_DIR}
    )
    target_link_libraries(tokenizer
        PRIVATE
        re2::re2
    )
    target_sources(tokenizer
        PRIVATE
        ${LLAMA2_TOKENIZER_DIR}/tiktoken.cpp
        ${LLAMA2_TOKENIZER_DIR}/bpe_tokenizer.cpp
    )

    # Include directory for neuron headers
    include_directories(
        BEFORE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/mediatek/runtime/include
    )

    # Build Llama Executor static library
    add_subdirectory(executor_runner/llama_runner)

    # Build Llama Executor Runner
    add_executable(mtk_llama_executor_runner ${_mtk_llama_executor_runner__srcs})

    target_link_libraries(mtk_llama_executor_runner
        ${_executor_runner_libs}
        neuron_backend
        gflags
        mtk_llama_executor_lib
        tokenizer
    )
    target_compile_options(mtk_llama_executor_runner
        PUBLIC
        ${_common_compile_options}
    )
endif()
