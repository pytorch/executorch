load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "irunner",
        exported_headers = [
            "irunner.h",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "constants",
        exported_headers = [
            "constants.h",
        ],
        visibility = ["PUBLIC"],
    )

    for aten in get_aten_mode_options():
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "stats" + aten_suffix,
            exported_headers = [
                "stats.h",
                "util.h",
            ],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":constants",
                 "//executorch/extension/module:module" + aten_suffix,
                 "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "text_decoder_runner" + aten_suffix,
            exported_headers = ["text_decoder_runner.h"],
            srcs = ["text_decoder_runner.cpp"],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":stats" + aten_suffix,
                "//executorch/kernels/portable/cpu/util:arange_util" + aten_suffix,
                "//executorch/extension/llm/sampler:sampler" + aten_suffix,
                "//executorch/extension/llm/runner/io_manager:io_manager" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "text_prefiller" + aten_suffix,
            exported_headers = ["text_prefiller.h"],
            srcs = ["text_prefiller.cpp"],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                "//pytorch/tokenizers:headers",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "text_token_generator" + aten_suffix,
            exported_headers = ["text_token_generator.h"],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                "//pytorch/tokenizers:headers",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "image_prefiller" + aten_suffix,
            exported_headers = ["image_prefiller.h", "image.h"],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":constants",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/extension/llm/sampler:sampler" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "multimodal_runner_lib" + aten_suffix,
            exported_headers = [
                "audio.h",
                "image.h",
                "wav_loader.h",
                "multimodal_input.h",
                "multimodal_runner.h",
                "multimodal_prefiller.h",
                "multimodal_decoder_runner.h",
            ],
            srcs = [
                "multimodal_prefiller.cpp",
            ],
            exported_deps = [
                ":text_decoder_runner" + aten_suffix,
                ":text_prefiller" + aten_suffix,
                ":image_prefiller" + aten_suffix,
                ":text_token_generator" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "runner_lib" + aten_suffix,
            exported_headers = [
                "text_llm_runner.h",
                "llm_runner_helper.h",
                "constants.h",
            ],
            srcs = [
                "text_llm_runner.cpp",
                "llm_runner_helper.cpp",
                "multimodal_runner.cpp",
            ],
            visibility = ["PUBLIC"],
            compiler_flags = [
                "-Wno-missing-prototypes",
            ],
            exported_deps = [
                ":image_prefiller" + aten_suffix,
                ":irunner",
                ":multimodal_runner_lib" + aten_suffix,
                ":text_decoder_runner" + aten_suffix,
                ":text_prefiller" + aten_suffix,
                ":text_token_generator" + aten_suffix,
                "//executorch/extension/llm/runner/io_manager:io_manager" + aten_suffix,
                "//pytorch/tokenizers:hf_tokenizer",
                "//pytorch/tokenizers:llama2c_tokenizer",
                "//pytorch/tokenizers:sentencepiece",
                "//pytorch/tokenizers:tekken",
                "//pytorch/tokenizers:tiktoken",
            ],
        )
