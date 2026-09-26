load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    for aten in get_aten_mode_options():
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "seq2seq_runner" + aten_suffix,
            exported_headers = [
                "runner.h",
                "seq2seq_runner.h",
            ],
            srcs = [
                "seq2seq_runner.cpp",
            ],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/extension/llm/runner:runner_lib" + aten_suffix,
                "//executorch/extension/llm/runner:stats" + aten_suffix,
                "//executorch/extension/llm/sampler:sampler" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//pytorch/tokenizers:headers",
            ],
        )

        runtime.cxx_library(
            name = "transducer_runner" + aten_suffix,
            exported_headers = [
                "transducer_runner.h",
            ],
            srcs = [
                "transducer_runner.cpp",
            ],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/extension/llm/runner:runner_lib" + aten_suffix,
                "//executorch/extension/llm/runner:stats" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//pytorch/tokenizers:headers",
            ],
        )
