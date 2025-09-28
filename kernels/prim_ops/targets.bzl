load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Define the filegroup once outside the loop since it doesn't vary by aten mode
    runtime.filegroup(
        name = "prim_ops_sources",
        srcs = ["register_prim_ops.cpp"],
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    runtime.filegroup(
        name = "selective_build_prim_ops.h",
        srcs = ["selective_build_prim_ops.h"],
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "et_copy_index" + aten_suffix,
            srcs = ["et_copy_index.cpp"],
            # To allow for selective prim ops to depend on this library.
            # Used by selective_build.bzl
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_headers = ["et_copy_index.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
                "//executorch/runtime/core:core",
            ],
            exported_deps = [
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "et_view" + aten_suffix,
            srcs = ["et_view.cpp"],
            # To allow for selective prim ops to depend on this library.
            # Used by selective_build.bzl
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_headers = ["et_view.h"],
            deps = [
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
                "//executorch/runtime/core:core",
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
            compiler_flags = select({
                "DEFAULT": ["-Wno-global-constructors"],
                "ovr_config//os:windows": [],
            }),
            deps = [
                ":et_copy_index" + aten_suffix,
                ":et_view" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/kernel:operator_registry" + aten_suffix,
                "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
            ],
        )
