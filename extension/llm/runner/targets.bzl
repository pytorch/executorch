load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "stats",
        exported_headers = ["stats.h"],
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
            ]
        )
