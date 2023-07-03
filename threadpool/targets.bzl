load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "threadpool",
        srcs = [
            "threadpool.cpp",
            "threadpool_guard.cpp",
            "fb/thread_pool_use_nthreads.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
        fbcode_deps = [
            ":cpuinfo",
        ],
        xplat_deps = [
            "//third-party/cpuinfo:cpuinfo",
        ],
        exported_headers = [
            "threadpool.h",
            "threadpool_guard.h",
            "fb/thread_pool_use_nthreads.h",
        ],
        exported_deps = [
            "//xplat/third-party/pthreadpool:pthreadpool",
        ],
        exported_preprocessor_flags = [
            "-DET_USE_THREADPOOL",
        ],
        visibility = [
            "//executorch/...",
            "//executorch/backends/...",
            "//executorch/threadpool/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
