load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        # Depend on this target if your types (Tensor, ArrayRef, etc) should be flexible between ATen and executor
        runtime.cxx_library(
            name = "lib" + aten_suffix,
            exported_headers = ["exec_aten.h"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = ["//executorch/runtime/core:tensor_shape_dynamism"] + ([] if aten_mode else ["//executorch/runtime/core/portable_type:portable_type"]),
            exported_external_deps = ["libtorch"] if aten_mode else [],
        )
