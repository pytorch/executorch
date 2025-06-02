load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_executorch_supported_platforms", "runtime")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

PLATFORMS = get_executorch_supported_platforms()

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
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "regex",
        srcs = [
            "src/re2_regex.cpp",
            "src/regex.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        exported_external_deps = [
            "re2",
        ],
        visibility = ["//pytorch/tokenizers/..."],
        header_namespace = "",
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "regex_lookahead",
        srcs = [
            "src/pcre2_regex.cpp",
            "src/regex_lookahead.cpp",
            "src/std_regex.cpp",
        ],
        exported_deps = [
            ":regex",
            ":headers",
        ],
        compiler_flags = [
            "-Wno-global-constructors",
            "-Wno-missing-prototypes",
        ],
        exported_external_deps = [
            "pcre2",
        ],
        # Making sure this library is not being stripped by linker.
        # @lint-ignore BUCKLINT: Avoid link_whole=True
        link_whole = True,
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        header_namespace = "",
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "bpe_tokenizer_base",
        srcs = [
            "src/bpe_tokenizer_base.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        exported_external_deps = [
            "re2",
        ],
        visibility = [
            "//pytorch/tokenizers/...",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "sentencepiece",
        srcs = [
            "src/sentencepiece.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        external_deps = [
            "sentencepiece",
            "abseil-cpp",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "tiktoken",
        srcs = [
            "src/tiktoken.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":bpe_tokenizer_base",
            ":headers",
        ],
        exported_external_deps = [
            "re2",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "hf_tokenizer",
        srcs = [
            "src/hf_tokenizer.cpp",
            "src/pre_tokenizer.cpp",
            "src/token_decoder.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":bpe_tokenizer_base",
            ":headers",
            "//pytorch/tokenizers/third-party:unicode",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        exported_external_deps = [
            "re2",
            "nlohmann_json",
        ],
        platforms = PLATFORMS,
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
        platforms = PLATFORMS,
    )
