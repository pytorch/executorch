load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def _get_operator_lib(aten = False):
    if aten:
        return ["//executorch/kernels/aten:generated_lib"]
    else:
        return [
            "//executorch/configurations:optimized_native_cpu_ops",
            "//executorch/extension/llm/custom_ops:custom_ops",
        ]

def define_common_targets():
    for aten in get_aten_mode_options():
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "parakeet_utils" + aten_suffix,
            srcs = [
                "timestamp_utils.cpp",
                "tokenizer_utils.cpp",
            ],
            exported_headers = [
                "timestamp_utils.h",
                "tokenizer_utils.h",
                "types.h",
            ],
            preprocessor_flags = [] if runtime.is_oss else ["-DEXECUTORCH_FB_BUCK"],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//pytorch/tokenizers:headers",
                "//pytorch/tokenizers/third-party:unicode",
                "//executorch/runtime/platform:platform",
            ],
        )

        runtime.cxx_binary(
            name = "parakeet_runner" + aten_suffix,
            srcs = ["main.cpp"],
            compiler_flags = ["-Wno-global-constructors"],
            preprocessor_flags = [] if runtime.is_oss else ["-DEXECUTORCH_FB_BUCK"],
            deps = [
                ":parakeet_utils" + aten_suffix,
                "//executorch/extension/llm/runner:stats" + aten_suffix,
                "//executorch/extension/llm/runner:runner_lib" + aten_suffix,
                "//executorch/extension/llm/runner:multimodal_runner_lib" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
                "//executorch/runtime/platform:platform",
                "//pytorch/tokenizers:sentencepiece",
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/backends/vulkan:vulkan_backend_lib",
                "//executorch/kernels/quantized:generated_lib" + aten_suffix,
            ] + _get_operator_lib(aten),
            external_deps = [
                "gflags",
            ] + (["libtorch"] if aten else []),
        )
