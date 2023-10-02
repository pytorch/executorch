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
        deps = [
            "//executorch/examples/models:models",  # @manual
        ],
        visibility = [
            "//executorch/examples/...",
        ],
    )

    runtime.python_binary(
        name = "aot_compiler",
        srcs = [
            "aot_compiler.py",
        ],
        main_module = "executorch.examples.recipes.xnnpack.aot_compiler",
        deps = [
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
            "//executorch/examples/recipes/xnnpack:models",
            "//executorch/examples/quantization/quant_flow:quant_utils",
            "//executorch/examples/export:utils",
            "//executorch/exir:lib",
            "//executorch/exir/backend:backend_api",
        ],
    )
