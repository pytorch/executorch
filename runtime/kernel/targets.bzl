load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _operator_registry_preprocessor_flags():
    max_kernel_num = native.read_config("executorch", "max_kernel_num", None)
    if max_kernel_num != None:
        return ["-DMAX_KERNEL_NUM=" + max_kernel_num]
    elif not runtime.is_oss:
        return select({
            "DEFAULT": [],
            "fbsource//xplat/executorch/build/constraints:executorch-max-kernel-num-256": ["-DMAX_KERNEL_NUM=256"],
            "fbsource//xplat/executorch/build/constraints:executorch-max-kernel-num-64": ["-DMAX_KERNEL_NUM=64"],
        })
    else:
        return []

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
        preprocessor_flags = _operator_registry_preprocessor_flags(),
    )

    runtime.cxx_library(
        name = "operator_registry_MAX_NUM_KERNELS_TEST_ONLY",
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
        preprocessor_flags = ["-DMAX_KERNEL_NUM=1"],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "kernel_runtime_context" + aten_suffix,
            exported_headers = [
                "kernel_runtime_context.h",
            ],
            visibility = [
                "//executorch/kernels/...",
                "//executorch/runtime/executor/...",
                "//executorch/runtime/kernel/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/platform:platform",
                "//executorch/runtime/core:memory_allocator",
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                # TODO(T147221312): This will eventually depend on exec_aten
                # once KernelRuntimeContext support tensor resizing, which is
                # why this target supports aten mode.
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
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
        )
