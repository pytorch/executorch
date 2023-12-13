# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# qnn_executor_runner: Like executor_runner but with QNN
if(${ANDROID})
    get_filename_component(EXECUTORCH_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../.."
        ABSOLUTE
    )
    set(_qnn_executor_runner__srcs ${_executor_runner__srcs})
    list(
        TRANSFORM
        _qnn_executor_runner__srcs
        PREPEND
        "${EXECUTORCH_SOURCE_DIR}/"
    )
    list(
        FILTER
        _qnn_executor_runner__srcs
        EXCLUDE REGEX
        ".*executor_runner.cpp$"
    )
    list(
        PREPEND
        _qnn_executor_runner__srcs
        ${CMAKE_CURRENT_LIST_DIR}/executor_runner/qnn_executor_runner.cpp
    )

    add_executable(qnn_executor_runner ${_qnn_executor_runner__srcs})

    target_link_libraries(qnn_executor_runner
        ${_executor_runner_libs}
        qnn_executorch_backend
        gflags
    )
    target_compile_options(qnn_executor_runner
        PUBLIC
        ${_common_compile_options}
    )
endif()
