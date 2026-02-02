load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def _get_operator_lib(aten = False):
    if aten:
        return ["//executorch/kernels/aten:generated_lib"]
    else:
        return ["//executorch/configurations:optimized_native_cpu_ops", "//executorch/extension/llm/custom_ops:custom_ops"]

def get_qnn_dependency():
    # buck build -c executorch.enable_qnn=true //executorch/examples/models/llama/runner:runner
    # Check if QNN is enabled before including the dependency
    if native.read_config("executorch", "enable_qnn", "false") == "true":
        # //executorch/backends/qualcomm:qnn_executorch_backend doesn't work,
        #  likely due to it's an empty library with dependency only
        return [
            "//executorch/backends/qualcomm/runtime:runtime",
        ]
    return []

def define_common_targets():
    for aten in get_aten_mode_options():
        aten_suffix = "_aten" if aten else ""
        runtime.cxx_library(
            name = "runner" + aten_suffix,
            srcs = [
                "runner.cpp",
            ],
            exported_headers = [
                "runner.h",
            ],
            deps = [
                "//executorch/devtools/etdump:etdump_flatcc",
            ],
            preprocessor_flags = [
                "-DUSE_ATEN_LIB",
            ] if aten else [],
            visibility = ["PUBLIC"],
            compiler_flags = [
                "-Wno-missing-prototypes",
            ],
            exported_deps = [
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/llm/runner:runner_lib" + aten_suffix,
                "//executorch/kernels/quantized:generated_lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/examples/models/llama/tokenizer:tiktoken",
                "//pytorch/tokenizers:llama2c_tokenizer",
                "//pytorch/tokenizers:hf_tokenizer",
            ] + (_get_operator_lib(aten)) + ([
                # Vulkan API currently cannot build on some platforms (e.g. Apple, FBCODE)
                # Therefore enable it explicitly for now to avoid failing tests
                "//executorch/backends/vulkan:vulkan_backend_lib",
            ] if native.read_config("llama", "use_vulkan", "0") == "1" else []) + get_qnn_dependency(),
            external_deps = [
                "libtorch",
            ] if aten else [],
        )

    runtime.cxx_library(
        name = "static_attention_io_manager",
        exported_headers = [
            "static_attention_io_manager.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/executor:program",
        ]
    )
