load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_library(
        name = "tokenizer_py_lib",
        srcs = [
            "__init__.py",
            "tokenizer.py",
        ],
        base_module = "executorch.extension.llm.tokenizer",
        visibility = [
            "//executorch/examples/...",
            "//bento/...",
            "//bento_kernels/...",
        ],
        _is_external_target = True,
        deps = [] if runtime.is_oss else ["fbsource//third-party/pypi/sentencepiece:sentencepiece"],
    )

    runtime.python_binary(
        name = "tokenizer_py",
        main_module = "executorch.extension.llm.tokenizer.tokenizer",
        visibility = [
            "//executorch/examples/...",
            "fbsource//xplat/executorch/examples/...",
        ],
        _is_external_target = True,
        deps = [
            ":tokenizer_py_lib",
        ],
    )
