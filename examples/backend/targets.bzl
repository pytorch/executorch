load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_binary(
        name = "xnnpack_examples",
        main_module = "executorch.examples.backend.xnnpack_examples",
        preload_deps = ["//executorch/kernels/quantized:aot_lib"],
        deps = [
            ":xnnpack_examples_lib",
        ],
    )

    runtime.python_library(
        name = "xnnpack_examples_lib",
        srcs = [
            "xnnpack_examples.py",
        ],
        deps = [
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
            "//executorch/examples/export:lib",
            "//executorch/examples/recipes/xnnpack_optimization:models",
            "//executorch/examples/quantization:quant_utils",
            "//executorch/exir:lib",
            "//executorch/exir/backend:backend_api",
        ],
    )

    # executor_runner for XNNPACK Backend and portable kernels.
    runtime.cxx_binary(
        name = "xnn_executor_runner",
        srcs = [],
        deps = [
            "//executorch/examples/executor_runner:executor_runner_lib",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/kernels/portable:generated_lib_all_ops",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
    )
