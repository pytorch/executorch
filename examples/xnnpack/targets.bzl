load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "models",
        srcs = [
            "__init__.py",
        ],
        visibility = [
            "//executorch/examples/xnnpack/...",
        ],
        deps = [
            "//executorch/examples/models:models",  # @manual
        ],
    )

    runtime.python_library(
        name = "xnnpack_aot_lib",
        srcs = [
            "aot_compiler.py",
        ],
        deps = [
            ":models",
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
            "//executorch/extension/export_util:export_util",
            "//executorch/examples/xnnpack/quantization:quant_utils",
            "//executorch/exir:lib",
            "//executorch/exir/backend:backend_api",
            "//executorch/devtools:lib",
        ],
    )

    runtime.python_binary(
        name = "aot_compiler",
        main_module = "executorch.examples.xnnpack.aot_compiler",
        resources = {
            "//executorch/examples/models/llama2/params:params": "params",
        },
        deps = [
            ":xnnpack_aot_lib",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # executor_runner for XNNPACK Backend and portable kernels.
    runtime.cxx_binary(
        name = "xnn_executor_runner",
        deps = [
            "//executorch/examples/portable/executor_runner:executor_runner_lib",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/kernels/portable:generated_lib",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
    )
