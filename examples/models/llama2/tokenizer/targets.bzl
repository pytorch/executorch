load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "bpe_tokenizer",
        srcs = [
            "bpe_tokenizer.cpp",
        ],
        exported_headers = [
            "tokenizer.h",
            "bpe_tokenizer.h",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "tiktoken",
        srcs = [
            "tiktoken.cpp",
            "llama_tiktoken.cpp",
        ],
        exported_headers = [
            "tokenizer.h",
            "tiktoken.h",
            "llama_tiktoken.h",
            "base64.h",
        ],
        exported_deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_external_deps = [
            "re2",
        ],
    )
