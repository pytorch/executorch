load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""

        exported_preprocessor_flags_ = []
        exported_deps_ = ["//executorch/runtime/core:core"]
        if aten_mode:
            exported_preprocessor_flags_ += ["-DUSE_ATEN_LIB"]
        else:
            exported_deps_ += ["//executorch/runtime/core/portable_type:scalar_type"]

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
            exported_preprocessor_flags = exported_preprocessor_flags_,
            exported_deps = exported_deps_,
            exported_external_deps = ["libtorch"] if aten_mode else [],
        )

        runtime.cxx_library(
            name = "dim_order_util" + aten_suffix,
            srcs = [],
            exported_headers = [
                "dim_order_util.h",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core/portable_type/c10/c10:c10",
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
                ":tensor_dimension_limit",
                "//executorch/runtime/core:core",
                "//executorch/runtime/core/portable_type/c10/c10:c10",
            ] + [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                ":scalar_type_util" + aten_suffix,
                ":dim_order_util" + aten_suffix,
            ],
            # WARNING: using a deprecated API to avoid being built into a shared
            # library. In the case of dynamically loading so library we don't want
            # it to depend on other so libraries because that way we have to
            # specify library directory path.
            force_static = True,
        )

        runtime.cxx_library(
            name = "tensor_shape_to_c_string" + aten_suffix,
            srcs = ["tensor_shape_to_c_string.cpp"],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core/exec_aten/util:tensor_dimension_limit",
            ],
            exported_headers = ["tensor_shape_to_c_string.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

    runtime.cxx_library(
        name = "tensor_dimension_limit",
        exported_headers = ["tensor_dimension_limit.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
