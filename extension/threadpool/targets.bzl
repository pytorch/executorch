load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    _THREADPOOL_SRCS = [
        "threadpool.cpp",
        "threadpool_guard.cpp",
    ] + (["fb/threadpool_use_n_threads.cpp"] if not runtime.is_oss else [])

    _THREADPOOL_HEADERS = [
        "threadpool.h",
        "threadpool_guard.h",
    ] + (["fb/threadpool_use_n_threads.h"] if not runtime.is_oss else [])

    runtime.cxx_library(
        name = "threadpool",
        srcs = _THREADPOOL_SRCS,
        deps = [
            "//executorch/runtime/core:core",
        ],
        exported_headers = _THREADPOOL_HEADERS,
        exported_deps = [
            third_party_dep("pthreadpool"),
            third_party_dep("cpuinfo"),
        ],
        exported_preprocessor_flags = [
            "-DET_USE_THREADPOOL",
        ],
        visibility = [
            "//executorch/...",
            "//executorch/backends/...",
            "//executorch/runtime/backend/...",
            "//executorch/extension/threadpool/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "cpuinfo_utils",
        srcs = [
            "cpuinfo_utils.cpp",
        ],
        deps = [
            "//executorch/runtime/core:core",
        ],
        exported_headers = [
            "cpuinfo_utils.h",
        ],
        exported_deps = [
            third_party_dep("pthreadpool"),
            third_party_dep("cpuinfo"),
        ],
        visibility = [
            "//executorch/...",
            "//executorch/backends/...",
            "//executorch/runtime/backend/...",
            "//executorch/extension/threadpool/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
