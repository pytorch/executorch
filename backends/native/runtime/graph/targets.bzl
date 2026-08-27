load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "format",
        exported_headers = [
            "Format.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "scalar_type",
        exported_headers = [
            "ScalarType.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "scalar",
        srcs = ["Scalar.cpp"],
        exported_headers = [
            "Scalar.h",
        ],
        deps = [":format"],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "tensor_meta",
        srcs = ["TensorMeta.cpp"],
        exported_headers = [
            "TensorMeta.h",
        ],
        exported_deps = [
            ":scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "ids",
        exported_headers = [
            "Ids.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "value",
        srcs = ["Value.cpp"],
        exported_headers = [
            "Value.h",
        ],
        exported_deps = [
            ":ids",
            ":scalar",
            ":tensor_meta",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    runtime.cxx_library(
        name = "argument",
        srcs = ["Argument.cpp"],
        exported_headers = [
            "Argument.h",
        ],
        exported_deps = [
            ":ids",
            ":scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
