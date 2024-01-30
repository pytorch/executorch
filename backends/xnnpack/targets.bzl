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
        ]),
        headers = native.glob([
            "runtime/*.h",
        ]),
        visibility = [
            "//executorch/exir/backend:backend_lib",
            "//executorch/exir/backend/test/...",
            "//executorch/backends/xnnpack/test/...",
            "//executorch/extension/pybindings/...",
            "@EXECUTORCH_CLIENTS",
        ],
        preprocessor_flags = [
            # "-DENABLE_XNNPACK_PROFILING",
        ] + ([] if runtime.is_oss else ["-DENABLE_DYNAMIC_QUANTIZATION"]),
        deps = [
            third_party_dep("XNNPACK"),
            ":dynamic_quant_utils",  # TODO Use (1) portable for choose_qparams(), (2) xnnpack for quantize_per_tensor(),
            "//executorch/runtime/backend:interface",
            "//executorch/backends/xnnpack/serialization:xnnpack_flatbuffer_header",
            "//executorch/backends/xnnpack/threadpool:threadpool",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        # XnnpackBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
