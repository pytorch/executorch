load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "tensor_util" + (aten_suffix),
            srcs = ["tensor_util.cpp"],
            exported_headers = [
                "tensor_util.h",
                "tensor_factory.h",
            ],
            visibility = ["PUBLIC"],
            compiler_flags = ["-Wno-unneeded-internal-declaration"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_dimension_limit",
                "//executorch/runtime/core/portable_type/c10/c10:c10",
            ],
            exported_external_deps = [
                "gmock" + aten_suffix,
            ] + (["libtorch"] if aten_mode else []),
        )
