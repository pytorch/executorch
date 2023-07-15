load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "operator_registry",
        srcs = ["operator_registry.cpp"],
        exported_headers = ["operator_registry.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "kernel_runtime_context" + aten_suffix,
            exported_headers = [
                "kernel_runtime_context.h",
            ],
            visibility = [
                "//executorch/kernels/prim_ops/...",  # Contains kernels
                "//executorch/runtime/kernel/...",
                "//executorch/kernels/...",
                "//executorch/runtime/executor/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "kernel_includes" + aten_suffix,
            exported_headers = [
                "kernel_includes.h",
            ],
            visibility = [
                "//executorch/runtime/kernel/...",
                "//executorch/kernels/...",
                "//executorch/kernels/prim_ops/...",  # Prim kernels
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":kernel_runtime_context" + aten_suffix,
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
                "//executorch/core/kernel_types/util:scalar_type_util" + aten_suffix,
                "//executorch/core/kernel_types/util:tensor_util" + aten_suffix,
            ],
        )
