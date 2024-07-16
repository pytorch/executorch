load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_test(
        name = "test_tokenizer_py",
        srcs = [
            "test_tokenizer.py",
        ],
        deps = [
            "//executorch/extension/llm/tokenizer:tokenizer_py_lib",
        ],
    )
