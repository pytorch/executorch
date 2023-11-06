load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    et_operator_library(
        name = "two_ops",
        ops = [
            "aten::add.out",
            "aten::mm.out",
        ],
    )

    executorch_generated_lib(
        name = "two_ops_lib",
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        kernel_deps = ["//executorch/kernels/portable:operators"],
        deps = [":two_ops"],
    )

    runtime.cxx_library(
        name = "scalar_type_util_TEST_ONLY",
        srcs = [],
        exported_headers = [
            "scalar_type_util.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
        exported_deps = [
            "//executorch/runtime/core/portable_type:scalar_type",
            ":two_ops_lib_headers",
        ],
    )

    dtype_selective_build_lib = native.read_config("executorch", "dtype_selective_build_lib", None)
    if dtype_selective_build_lib != None:
        # retrieve selected_op_variants.h from codegen
        genrule_name = dtype_selective_build_lib + "_et_op_dtype_gen[selected_op_variants]"
        runtime.cxx_library(
            name = "dtype_headers",
            srcs = [],
            exported_headers = {
                "selected_op_variants.h": genrule_name,
            },
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        exported_preprocessor_flags_ = []
        exported_deps_ = ["//executorch/runtime/core:core"]
        if aten_mode:
            exported_preprocessor_flags_ += ["-DUSE_ATEN_LIB"]
        else:
            exported_deps_ += ["//executorch/runtime/core/portable_type:scalar_type"]

        if dtype_selective_build_lib != None:
            exported_preprocessor_flags_ += ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"]
            exported_deps_ += [":dtype_headers"]

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
            # WARNING: using a deprecated API to avoid being built into a shared
            # library. In the case of dynamically loading so library we don't want
            # it to depend on other so libraries because that way we have to
            # specify library directory path.
            force_static = True,
        )
