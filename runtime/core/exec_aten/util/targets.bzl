load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_xplat")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # get deps for kernel_types
    if is_xplat():
        aten_types_deps = ["//xplat/caffe2:torch_mobile_core"]
    else:
        aten_types_deps = ["//caffe2:torch-cpp"]

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "scalar_type_util" + aten_suffix,
            srcs = [],
            exported_headers = [
                "scalar_type_util.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/core:core",
            ] + aten_types_deps if aten_mode else ["//executorch/runtime/core/portable_type:scalar_type"],
        )

        runtime.cxx_library(
            name = "dim_order_util" + aten_suffix,
            srcs = [],
            exported_headers = [
                "dim_order_util.h",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
        )

        runtime.cxx_library(
            name = "tensor_util" + aten_suffix,
            srcs = ["tensor_util_aten.cpp"] if aten_mode else ["tensor_util_portable.cpp"],
            exported_headers = [
                "tensor_util.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/core:core",
            ] + [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                ":scalar_type_util" + aten_suffix,
                ":dim_order_util" + aten_suffix,
            ],
        )
