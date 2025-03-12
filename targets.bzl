load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "headers",
        exported_headers = subdir_glob([
            ("include", "pytorch/tokenizers/*.h"),
        ]),
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        header_namespace = "",
    )

    runtime.cxx_library(
        name = "sentencepiece",
        srcs = [
            "src/sentencepiece.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        compiler_flags = [
            "-D_USE_INTERNAL_STRING_VIEW",
        ],
        external_deps = [
            "sentencepiece",
            "abseil-cpp",
        ],
    )

    runtime.cxx_library(
        name = "tiktoken",
        srcs = [
            "src/tiktoken.cpp",
            "src/bpe_tokenizer_base.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        compiler_flags = [
            "-D_USE_INTERNAL_STRING_VIEW",
        ],
        exported_external_deps = [
            "re2",
        ],
    )

    runtime.cxx_library(
        name = "unicode",
        srcs = [
            "third-party/llama.cpp-unicode/src/unicode.cpp",
            "third-party/llama.cpp-unicode/src/unicode-data.cpp",
        ],
        exported_headers = subdir_glob([
            ("include", "pytorch/tokenizers/third-party/llama.cpp-unicode/*.h"),
        ]),
        header_namespace = "",
    )

    runtime.cxx_library(
        name = "hf_tokenizer",
        srcs = [
            "src/hf_tokenizer.cpp",
            "src/bpe_tokenizer_base.cpp",
            "src/pre_tokenizer.cpp",
            "src/token_decoder.cpp",
        ],
        exported_deps = [
            ":headers",
            ":unicode",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        compiler_flags = [
            "-D_USE_INTERNAL_STRING_VIEW",
        ],
        exported_external_deps = [
            "re2",
            "nlohmann_json",
        ],
    )

    runtime.cxx_library(
        name = "llama2c_tokenizer",
        srcs = [
            "src/llama2c_tokenizer.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
    )
