load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Debug-dump formatting shared by every renderer of the IR (header-only).
    runtime.cxx_library(
        name = "format",
        exported_headers = [
            "Format.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # Scalar element type + C++-type mapping (header-only; macro-driven, standalone).
    runtime.cxx_library(
        name = "scalar_type",
        exported_headers = [
            "ScalarType.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # A concrete scalar value (int / double / bool), tagged. c10::Scalar analog.
    runtime.cxx_library(
        name = "scalar",
        srcs = ["Scalar.cpp"],
        exported_headers = [
            "Scalar.h",
        ],
        deps = [":format"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Concrete in-memory IR value types (pure std; no ExecuTorch, no flatbuffers).
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
