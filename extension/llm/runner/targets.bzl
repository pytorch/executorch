load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "stats",
        exported_headers = [
            "stats.h",
            "util.h",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    for aten in (True, False):
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "text_decoder_runner" + aten_suffix,
            exported_headers = ["text_decoder_runner.h"],
            srcs = ["text_decoder_runner.cpp"],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":stats",
                "//executorch/extension/llm/sampler:sampler" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/runner_util:managed_tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "text_prefiller" + aten_suffix,
            exported_headers = ["text_prefiller.h"],
            srcs = ["text_prefiller.cpp"],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                "//executorch/extension/llm/tokenizer:tokenizer_header",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/runner_util:managed_tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "text_token_generator" + aten_suffix,
            exported_headers = ["text_token_generator.h"],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                "//executorch/extension/llm/tokenizer:tokenizer_header",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/runner_util:managed_tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "runner_lib" + aten_suffix,
            exported_headers = [
                "image_prefiller.h",
                "image.h",
                "multimodal_runner.h",
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                ":text_prefiller" + aten_suffix,
                ":text_token_generator" + aten_suffix,
            ],
        )
