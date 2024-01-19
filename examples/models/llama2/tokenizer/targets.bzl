load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "tokenizer_lib",
        srcs = ["tokenizer.cpp"],
        headers = ["tokenizer.h"],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/kernel:kernel_includes",
        ],
        visibility = [
            "//executorch/...",
        ],
    )

    if not runtime.is_oss:
        # no resources support
        runtime.export_file(
            name = "tokenizer_file",
            src = "test/test.bin",
        )

        runtime.cxx_test(
            name = "test_tokenizer_cpp",
            srcs = ["test/test_tokenizer.cpp"],
            deps = [
                ":tokenizer_lib",
                "//executorch/codegen:macros",
                "fbsource//xplat/tools/cxx:resources",
            ],
            resources = [":tokenizer_file"],
        )
