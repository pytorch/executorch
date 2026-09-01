load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "string_format",
        exported_headers = [
            "StringFormat.h",
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

    # utils/ has no BUCK of its own, so the IR printer's target lives here. Kept
    # separate from the IR libraries so only a consumer that dumps the IR links
    # the formatting code.
    runtime.cxx_library(
        name = "print",
        srcs = ["utils/Print.cpp"],
        exported_headers = [
            "utils/Print.h",
        ],
        exported_deps = [
            ":scalar",
            ":tensor_meta",
        ],
        deps = [":string_format"],
        visibility = ["//executorch/backends/native/..."],
    )
