load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "test_base64",
        srcs = [
            "test_base64.cpp",
        ],
        deps = [
            "//pytorch/tokenizers:headers",
        ],
    )

    runtime.cxx_test(
        name = "test_llama2c_tokenizer",
        srcs = [
            "test_llama2c_tokenizer.cpp",
        ],
        deps = [
            "//pytorch/tokenizers:llama2c_tokenizer",
        ],
        env = {
            "RESOURCES_PATH": "$(location :resources)/resources",
        },
        platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
    )

    runtime.cxx_test(
        name = "test_pre_tokenizer",
        srcs = [
            "test_pre_tokenizer.cpp",
        ],
        deps = [
            "//pytorch/tokenizers:headers",
            "//pytorch/tokenizers:hf_tokenizer",
        ],
    )

    runtime.cxx_test(
        name = "test_sentencepiece",
        srcs = [
            "test_sentencepiece.cpp",
        ],
        deps = ["//pytorch/tokenizers:sentencepiece"],
        external_deps = [
            "sentencepiece",
            "abseil-cpp",
        ],
        env = {
            "RESOURCES_PATH": "$(location :resources)/resources",
        },
        platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
    )

    runtime.cxx_test(
        name = "test_tiktoken",
        srcs = [
            "test_tiktoken.cpp",
        ],
        deps = [
            "//pytorch/tokenizers:tiktoken",
        ],
        env = {
            "RESOURCES_PATH": "$(location :resources)/resources",
        },
        platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
        external_deps = [
            "re2",
        ],
    )

    runtime.filegroup(
        name = "resources",
        srcs = native.glob([
            "resources/**",
        ]),
    )
