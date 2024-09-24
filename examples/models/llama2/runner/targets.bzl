load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _get_operator_lib(aten = False):
    if aten:
        return ["//executorch/kernels/aten:generated_lib"]
    elif runtime.is_oss:
        # TODO(T183193812): delete this path after optimized-oss.yaml is no more.
        return ["//executorch/configurations:optimized_native_cpu_ops_oss", "//executorch/extension/llm/custom_ops:custom_ops"]
    else:
        return ["//executorch/configurations:optimized_native_cpu_ops", "//executorch/extension/llm/custom_ops:custom_ops"]

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
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            # qnn_executorch_backend can be added below //executorch/backends/qualcomm:qnn_executorch_backend
            exported_deps = [
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/llm/runner:stats",
                "//executorch/extension/llm/runner:text_decoder_runner" + aten_suffix,
                "//executorch/extension/llm/runner:text_prefiller" + aten_suffix,
                "//executorch/extension/llm/runner:text_token_generator" + aten_suffix,
                "//executorch/extension/evalue_util:print_evalue" + aten_suffix,
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/kernels/quantized:generated_lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/examples/models/llama2/tokenizer:tiktoken",
                "//executorch/extension/llm/tokenizer:bpe_tokenizer",
            ] + (_get_operator_lib(aten)) + ([
                # Vulkan API currently cannot build on some platforms (e.g. Apple, FBCODE)
                # Therefore enable it explicitly for now to avoid failing tests
                "//executorch/backends/vulkan:vulkan_backend_lib",
            ] if native.read_config("llama", "use_vulkan", "0") == "1" else []),
            external_deps = [
                "libtorch",
            ] if aten else [],
        )
