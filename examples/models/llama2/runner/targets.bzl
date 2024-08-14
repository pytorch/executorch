load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _get_operator_lib(aten = False):
    if aten:
        return ["//executorch/kernels/aten:generated_lib"]
    elif runtime.is_oss:
        return ["//executorch/kernels/portable:generated_lib", "//executorch/extension/llm/custom_ops:custom_ops"]
    else:
        return ["//executorch/configurations:optimized_native_cpu_ops", "//executorch/extension/llm/custom_ops:custom_ops"]

def use_tiktoken():
    return native.read_config("llama", "use_tiktoken", "0") == "1"

def define_common_targets():
    for aten in (True, False):
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "runner" + aten_suffix,
            srcs = [
                "runner.cpp",
            ],
            exported_headers = [
                "runner.h",
            ],
            preprocessor_flags = [
                "-DUSE_ATEN_LIB",
            ] if aten else [],
            exported_preprocessor_flags = ["-DET_USE_TIKTOKEN"] if use_tiktoken() else [],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/llm/runner:stats",
                "//executorch/extension/llm/runner:text_decoder_runner" + aten_suffix,
                "//executorch/extension/llm/runner:text_prefiller" + aten_suffix,
                "//executorch/extension/llm/runner:text_token_generator" + aten_suffix,
                "//executorch/extension/evalue_util:print_evalue" + aten_suffix,
                "//executorch/extension/runner_util:managed_tensor" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/kernels/quantized:generated_lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ] + ([
                "//executorch/examples/models/llama2/tokenizer:tiktoken",
            ] if use_tiktoken() else [
                "//executorch/extension/llm/tokenizer:bpe_tokenizer",
            ]) + (_get_operator_lib(aten)) + ([
                # Vulkan API currently cannot build on some platforms (e.g. Apple, FBCODE)
                # Therefore enable it explicitly for now to avoid failing tests
                "//executorch/backends/vulkan:vulkan_backend_lib",
            ] if native.read_config("llama", "use_vulkan", "0") == "1" else []),
            external_deps = [
                "libtorch",
            ] if aten else [],
        )
