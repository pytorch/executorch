"""Client build configurations.

This package contains useful build targets for executorch clients, assembling
common collections of deps into self-contained targets.
"""

load("@fbsource//xplat/executorch/backends:backends.bzl", "get_all_cpu_backend_targets")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # An extended executor library that includes all CPU backend targets and
    # helper deps.
    runtime.cxx_library(
        name = "executor_cpu_optimized",
        exported_deps = [
            "//executorch/extension/threadpool:threadpool",
        ] + get_all_cpu_backend_targets(),
        visibility = [
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Add a common configuration of cpu optimized operators. This adds a bit of confusion
    # with the above executorch_cpu_optimized target. Generally it would make sense
    # to just add optimized operators to that target but because executorch_cpu_optimized
    # might be used elsewhere, I dont want to include ops in that target and find out
    # size implication. Aim to simplify this later.
    # File a task, TODO(task), for llama work
    executorch_generated_lib(
        name = "optimized_native_cpu_ops",
        deps = [
            "//executorch/kernels/optimized:optimized_operators",
            "//executorch/kernels/optimized:optimized_oplist",
            "//executorch/kernels/portable:executorch_aten_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/optimized:optimized.yaml",
        fallback_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/examples/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # TODO(T183193812): delete this target after optimized-oss.yaml is gone
    executorch_generated_lib(
        name = "optimized_native_cpu_ops_oss",
        deps = [
            "//executorch/kernels/optimized:optimized_operators",
            "//executorch/kernels/optimized:optimized_oplist",
            "//executorch/kernels/portable:executorch_aten_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/optimized:optimized-oss.yaml",
        fallback_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
        visibility = [
            "//executorch/examples/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
