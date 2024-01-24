load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "test",
        srcs = [
            "test_tokenizer.cpp",
        ],
        deps = [
            "//executorch/examples/models/llama2/tokenizer:tokenizer",
        ],
        env = {
            "RESOURCES_PATH": "$(location :resources)/resources",
        },
    )

    runtime.filegroup(
        name = "resources",
        srcs = native.glob([
            "resources/**",
        ]),
    )

    runtime.python_test(
        name = "test_tokenizer_py",
        srcs = [
            "test_tokenizer.py",
        ],
        visibility = [
            "//executorch/examples/...",
        ],
        deps = [
            "//executorch/examples/models/llama2/tokenizer:tokenizer_py_lib",
        ],
    )
