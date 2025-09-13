load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:build_variables.bzl", "THREADPOOL_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    _THREADPOOL_SRCS = THREADPOOL_SRCS + (
        ["fb/threadpool_use_n_threads.cpp"] if not runtime.is_oss else []
    )

    _THREADPOOL_HEADERS = [
        "threadpool.h",
        "threadpool_guard.h",
    ] + (["fb/threadpool_use_n_threads.h"] if not runtime.is_oss else [])

    runtime.cxx_library(
        name = "threadpool_lib",
        srcs = _THREADPOOL_SRCS,
        deps = [
            ":cpuinfo_utils",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/portable_type/c10/c10:c10",
        ],
        exported_headers = _THREADPOOL_HEADERS,
        exported_deps = [
            third_party_dep("pthreadpool"),
            third_party_dep("cpuinfo"),
            # Allow users to use the header without an extra deps entry.
            "//executorch/runtime/kernel:thread_parallel_interface",
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
        name = "threadpool",
        # TODO: OSS doesn't have os:iphoneos. Sync buck2 prelude
        # update to add it and remove duplication.
        exported_deps = (select({
            # Major operating systems should be able to use threadpool.
            "ovr_config//os:linux": [":threadpool_lib"],
            "ovr_config//os:macos": [":threadpool_lib"],
            "ovr_config//os:windows": [":threadpool_lib"],
            "ovr_config//os:android": [":threadpool_lib"],
            "ovr_config//os:iphoneos": [":threadpool_lib"],
            # Machines without an operating system shouldn't.
            "ovr_config//os:none": ["//executorch/runtime/kernel:thread_parallel_interface"],
            # If we don't know what it is, disable threadpool out of caution.
            "DEFAULT": ["//executorch/runtime/kernel:thread_parallel_interface"],
        }) if not runtime.is_oss else select({
            # Major operating systems should be able to use threadpool.
            "ovr_config//os:linux": [":threadpool_lib"],
            "ovr_config//os:macos": [":threadpool_lib"],
            "ovr_config//os:windows": [":threadpool_lib"],
            "ovr_config//os:android": [":threadpool_lib"],
            # Machines without an operating system shouldn't.
            "ovr_config//os:none": ["//executorch/runtime/kernel:thread_parallel_interface"],
            # If we don't know what it is, disable threadpool out of caution.
            "DEFAULT": ["//executorch/runtime/kernel:thread_parallel_interface"],
        })),
        visibility = [
            "//executorch/...",
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
