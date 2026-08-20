load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/third-party:third_party_libs.bzl", "qnn_third_party_dep")
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

def define_common_targets(is_fbcode = False):
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


    if is_fbcode:
        runtime.python_library(
            name = "whisper_model_lib",
            srcs = [
                "whisper_model.py",
            ],
            deps = [
                "//caffe2:torch",
                "fbsource//third-party/pypi/transformers:transformers",
            ],
        )

        runtime.python_library(
            name = "whisper_lib",
            srcs = ["whisper.py"],
            deps = [
                ":whisper_model_lib",
                "//caffe2:torch",
                "//executorch/backends/qualcomm:export_utils",
                "//executorch/backends/qualcomm/_passes:passes",
                "//executorch/backends/qualcomm/partition:partition",
                "//executorch/backends/qualcomm/quantizer:quantizer",
                "//executorch/backends/qualcomm/serialization:serialization",
                "//executorch/backends/qualcomm/utils:utils",
                "//executorch/devtools/backend_debug:delegation_info",
                "//executorch/examples/qualcomm:utils",
                "//executorch/exir/capture:config",
                "//executorch/exir/passes:memory_planning_pass",
                "fbsource//third-party/pypi/datasets:datasets",
                "fbsource//third-party/pypi/librosa:librosa",
                "fbsource//third-party/pypi/soundfile:soundfile",
                "fbsource//third-party/pypi/torchmetrics:torchmetrics",
                "fbsource//third-party/pypi/transformers:transformers",
            ],
        )

        python_binary(
            name = "whisper",
            main_module = "executorch.examples.qualcomm.oss_scripts.whisper.whisper",
            deps = [
                ":whisper_lib",
            ],
        )
