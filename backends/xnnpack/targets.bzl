load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "dynamic_quant_utils",
        srcs = [
            "runtime/utils/utils.cpp",
        ],
        exported_headers = ["runtime/utils/utils.h"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/backend:interface",
        ],
        visibility = [
            "//executorch/backends/xnnpack/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "xnnpack_backend",
        srcs = native.glob([
            "runtime/*.cpp",
            "runtime/profiling/*.cpp",
        ]),
        headers = native.glob([
            "runtime/*.h",
            "runtime/profiling/*.h",
        ]),
        visibility = [
            "//executorch/exir/backend:backend_lib",
            "//executorch/exir/backend/test/...",
            "//executorch/backends/xnnpack/test/...",
            "//executorch/extension/pybindings/...",
            "@EXECUTORCH_CLIENTS",
        ],
        preprocessor_flags = [
            # Uncomment to enable per operator timings
            # "-DENABLE_XNNPACK_PROFILING",
            # Uncomment to enable workspace sharing across delegates
            # "-DENABLE_XNNPACK_SHARED_WORKSPACE"
        ],
        exported_deps = [
            "//executorch/runtime/backend:interface",
        ],
        deps = [
            third_party_dep("XNNPACK"),
            "//executorch/backends/xnnpack/serialization:xnnpack_flatbuffer_header",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        # XnnpackBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
