load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    _THREADPOOL_TESTS = [
        "threadpool_test.cpp",
    ] + (["fb/threadpool_use_n_threads_test.cpp"] if not runtime.is_oss else [])

    runtime.cxx_test(
        name = "threadpool_test",
        srcs = _THREADPOOL_TESTS,
        deps = [
            "//executorch/extension/threadpool:threadpool",
        ],
    )

    runtime.cxx_test(
        name = "thread_parallel_test",
        srcs = [
            "thread_parallel_test.cpp",
        ],
        deps = [
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/kernel:thread_parallel_interface",
            "//executorch/runtime/platform:platform",
        ],
    )
