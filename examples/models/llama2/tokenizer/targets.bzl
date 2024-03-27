load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "tokenizer",
        srcs = [
            "tokenizer.cpp",
        ],
        exported_headers = [
            "tokenizer.h",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.python_library(
        name = "tokenizer_py_lib",
        srcs = [
            "__init__.py",
            "tokenizer.py",
        ],
        base_module = "executorch.examples.models.llama2.tokenizer",
        visibility = [
            "//executorch/examples/...",
            "//bento/...",
        ],
        _is_external_target = True,
        deps = [] if runtime.is_oss else ["fbsource//third-party/pypi/sentencepiece:sentencepiece"],
    )

    runtime.python_binary(
        name = "tokenizer_py",
        main_module = "executorch.examples.models.llama2.tokenizer.tokenizer",
        visibility = [
            "//executorch/examples/...",
            "fbsource//xplat/executorch/examples/...",
        ],
        _is_external_target = True,
        deps = [
            ":tokenizer_py_lib",
        ],
    )
