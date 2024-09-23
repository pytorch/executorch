load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.python_library(
        name = "tokenizer_py_lib",
        srcs = [
            "__init__.py",
            "tokenizer.py",
            "utils.py",
        ],
        base_module = "executorch.extension.llm.tokenizer",
        visibility = [
            "//executorch/examples/...",
            "//executorch/extension/llm/tokenizer/...",
            "//executorch/extension/llm/export/...",
            "//bento/...",
            "//bento_kernels/...",
        ],
        _is_external_target = True,
        deps = [
            "//executorch/examples/models/llama2/tokenizer:tiktoken_py",
        ],
        external_deps = [
            "sentencepiece-py",
        ],
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

    runtime.cxx_library(
        name = "tokenizer_header",
        exported_headers = [
            "tokenizer.h",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "bpe_tokenizer",
        srcs = [
            "bpe_tokenizer.cpp",
        ],
        exported_headers = [
            "bpe_tokenizer.h",
        ],
        exported_deps = [
            ":tokenizer_header",
            "//executorch/runtime/core:core",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "tiktoken",
        srcs = [
            "tiktoken.cpp",
        ],
        exported_headers = [
            "tiktoken.h",
            "base64.h",
        ],
        exported_deps = [
            ":tokenizer_header",
            "//executorch/runtime/core:core",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_external_deps = [
            "re2",
        ],
    )
