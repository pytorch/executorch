load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "kernel_util",
        srcs = [],
        exported_headers = [
            "make_boxed_from_unboxed_functor.h",
            "meta_programming.h",
            "type_list.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/runtime/kernel:kernel_runtime_context",
            "//executorch/runtime/kernel:operator_registry",
        ],
    )
