load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "runner",
        srcs = ["llava_runner.cpp"],
        exported_headers = ["llava_runner.h", "llava_image_prefiller.h", "llava_text_decoder_runner.h"],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        compiler_flags = [
            "-Wno-global-constructors",
        ],
        exported_deps = [
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/extension/llm/tokenizer:bpe_tokenizer",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/kernels/quantized:generated_lib",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    )
