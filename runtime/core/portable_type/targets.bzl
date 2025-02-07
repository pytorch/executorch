load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Internal target for executor tensor. Clients should depend on
    # :kernel_types below to be flexible with ATen Tensor and executor Tensor.
    runtime.cxx_library(
        name = "portable_type",
        srcs = ["tensor_impl.cpp"],
        exported_headers = [
            "tensor_options.h",
            "optional.h",
            "scalar.h",
            "tensor.h",
            "tensor_impl.h",
            "string_view.h",
            "device.h",
        ],
        # Only should be depended on by kernel_types:kernel_types, but various suffixes like Android and Static
        # mean I cant just expose visibility to a single rule.
        visibility = [
            "//executorch/backends/...",
            "//executorch/runtime/core/exec_aten/...",
            "//executorch/runtime/core/portable_type/test/...",
        ],
        exported_deps = [
            ":scalar_type",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:tensor_shape_dynamism",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:dim_order_util",
            "//executorch/runtime/core:tag",
        ],
    )

    # Set up a specific exported library for scalar_type to avoid circle dependency in ScalarTypeUtil.h
    runtime.cxx_library(
        name = "scalar_type",
        exported_headers = [
            "bfloat16.h",
            "bfloat16_math.h",
            "complex.h",
            "half.h",
            "scalar_type.h",
            "qint_types.h",
            "bits_types.h",
        ],
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10:c10",
        ],
        visibility = [
            "//executorch/extension/...",
            "//executorch/runtime/core/exec_aten/util/...",
            "//executorch/kernels/...",
        ],
    )
