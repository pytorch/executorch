load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "et_copy_index" + aten_suffix,
            srcs = ["et_copy_index.cpp"],
            visibility = [],  # Private
            exported_headers = ["et_copy_index.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "et_view" + aten_suffix,
            srcs = ["et_view.cpp"],
            visibility = [],  # Private
            exported_headers = ["et_view.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "prim_ops_registry" + aten_suffix,
            srcs = ["register_prim_ops.cpp"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            # @lint-ignore BUCKLINT link_whole, need this to register prim ops.
            link_whole = True,
            # prim ops are registered through a global table so the ctor needs to be allowed
            compiler_flags = ["-Wno-global-constructors"],
            deps = [
                ":et_copy_index" + aten_suffix,
                ":et_view" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/kernel:operator_registry",
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
            ],
        )
