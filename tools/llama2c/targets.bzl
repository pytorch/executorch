load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "lib",
        srcs = [
            "__init__.py",
            "convert.py",
        ],
        deps = [
            "//pytorch/tokenizers/pytorch_tokenizers:tokenizers",
        ],
    )

    runtime.python_binary(
        name = "convert",
        main_module = "pytorch.tokenizers.tools.llama2c.convert",
        visibility = [
            "//executorch/examples/...",
            "fbsource//xplat/executorch/examples/...",
        ],
        _is_external_target = True,
        deps = [
            ":lib",
        ],
    )
