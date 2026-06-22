load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/third-party:third_party_libs.bzl", "qnn_third_party_dep")

def define_common_targets():
    runtime.cxx_library(
        name = "runner_lib",
        srcs = glob(
            [
                "runner/*.cpp",
            ],
        ),
        exported_headers = glob([
            "runner/*.h",
        ]),
        compiler_flags = [
            "-Wno-global-constructors",
            "-Wunused-command-line-argument",
        ],
        deps = [
            "//executorch/extension/llm/runner:stats",
            "//executorch/kernels/quantized:generated_lib",
            qnn_third_party_dep("api"),
        ],
        exported_deps = [
            "//executorch/extension/module:module",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//pytorch/tokenizers:hf_tokenizer",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/backends/qualcomm/runtime:runtime",
        ],
        external_deps = [
            "gflags",
        ],
        platforms = [ANDROID],
        **get_oss_build_kwargs()
    )

    runtime.cxx_binary(
        name = "qnn_whisper_runner",
        srcs = [
            "qnn_whisper_runner.cpp",
        ],
        compiler_flags = [
            "-Wno-global-constructors",
        ],
        deps = [
            ":runner_lib",
            "//executorch/extension/threadpool:threadpool",
        ],
        external_deps = [
            "gflags",
        ],
        platforms = [ANDROID],
        **get_oss_build_kwargs()
    )
