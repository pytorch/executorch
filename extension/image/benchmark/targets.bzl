load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_binary(
        name = "image_processor_benchmark",
        srcs = ["image_processor_benchmark.cpp"],
        deps = [
            "//executorch/extension/image:image_processor",
            "//executorch/extension/tensor:tensor",
        ],
    )

    runtime.python_library(
        name = "compare_benchmarks_lib",
        srcs = ["compare_benchmarks.py"],
        base_module = "",
    )

    runtime.python_binary(
        name = "compare_benchmarks",
        main_module = "compare_benchmarks",
        deps = [":compare_benchmarks_lib"],
    )
