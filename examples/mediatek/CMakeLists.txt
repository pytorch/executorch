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
endif()
