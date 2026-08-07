load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")

GEN_KERNEL_BACKEND_DEPS = [
    "//executorch/configurations:optimized_native_cpu_ops",
    "//executorch/kernels/quantized:generated_lib",
]

def _get_torchao_lowbit_deps():
    """Returns torchao lowbit kernel deps for shared embedding and linear on ARM builds."""
    return select({
        "DEFAULT": [],
        "ovr_config//cpu:arm64": [
            "//xplat/pytorch/ao/torchao/csrc/cpu/shared_kernels/embedding_xbit:op_embedding_xbit_executorch",
            "//xplat/pytorch/ao/torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight:op_linear_8bit_act_xbit_weight_executorch",
        ],
    })

def define_webgpu_python_targets():
    fbcode_target(_kind = runtime.python_library,
        name = "webgpu_support",
        srcs = [
            "mtp_qat_contract.py",
            "webgpu_artifact_manifest.py",
            "webgpu_partitioner.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.gemma4",
        resources = {
            "config/e2b_config.json": "config/e2b_config.json",
            "manifests/gemma4_e2b_webgpu.json": "manifests/gemma4_e2b_webgpu.json",
        },
        typing = True,
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:op_registry",
            "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
            "//executorch/backends/vulkan/patterns:vulkan_patterns",
            "//executorch/backends/webgpu/scripts:webgpu_artifact_manifest",
            "//executorch/examples/models/gemma4:target_prefill_contract",
            "//executorch/exir:lib",
        ],
    )

    fbcode_target(_kind = runtime.python_binary,
        name = "webgpu_artifact_manifest",
        main_function = "executorch.examples.models.gemma4.webgpu_artifact_manifest.main",
        deps = [":webgpu_support"],
    )

def define_common_targets():
    _KERNEL_BACKEND_DEPS = [
        "//executorch/backends/xnnpack:xnnpack_backend",
        "//executorch/extension/llm/custom_ops:custom_ops",
    ]

    # Shared C++ image preprocessing utilities (header-only)
    runtime.cxx_library(
        name = "image_utils_cpp",
        exported_headers = ["image_utils.h"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/extension/tensor:tensor",
        ],
    )

    # Gemma 4 runner library (reusable by CLI and JNI)
    runtime.cxx_library(
        name = "gemma4_runner",
        exported_headers = [
            "runner/gemma4_runner.h",
            "runner/gemma4_stats.h",
            "runner/generation_config.h",
        ],
        srcs = [
            "runner/gemma4_runner.cpp",
        ],
        visibility = ["PUBLIC"],
        deps = _KERNEL_BACKEND_DEPS + [
            "//executorch/backends/xnnpack:xnnpack_interface",
            "//executorch/runtime/backend:interface",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/llm/runner:runner_lib",
            "//pytorch/tokenizers:headers",
        ],
    )

    runtime.cxx_library(
        name = "gemma4_spec_runner",
        srcs = ["runner/gemma4_spec_runner.cpp"],
        exported_headers = ["runner/gemma4_spec_runner.h"],
        compiler_flags = ["-fexceptions"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
        deps = [
            "//executorch/backends/webgpu:webgpu_backend",
            "//executorch/backends/webgpu:webgpu_model_loader",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
        ],
    )

    runtime.cxx_library(
        name = "gemma4_spec_wasm_adapter",
        srcs = ["runner/gemma4_spec_wasm.cpp"],
        compiler_flags = ["-fexceptions"],
        visibility = ["PUBLIC"],
        deps = [
            ":gemma4_spec_runner",
            "//executorch/backends/webgpu:webgpu_backend",
        ],
    )

    runtime.cxx_binary(
        name = "gemma4_spec_runner_cli",
        srcs = ["runner/gemma4_spec_main.cpp"],
        compiler_flags = ["-fexceptions"],
        visibility = ["PUBLIC"],
        deps = [":gemma4_spec_runner"],
    )

    runtime.cxx_binary(
        name = "main",
        srcs = ["e2e_runner.cpp"],
        deps = [
            ":gemma4_runner",
            ":image_utils_cpp",
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
        ] + _KERNEL_BACKEND_DEPS + GEN_KERNEL_BACKEND_DEPS + _get_torchao_lowbit_deps(),
        external_deps = [
            "gflags",
            "stb",
        ],
        visibility = ["PUBLIC"],
        compiler_flags = ["-Wno-global-constructors"],
        preprocessor_flags = ["-DET_USE_THREADPOOL"],
    )

    runtime.cxx_binary(
        name = "gemma4_plain_wasm",
        srcs = ["runner/gemma4_plain_wasm.cpp"],
        compiler_flags = ["-fexceptions"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/backends/webgpu:webgpu_backend",
            "//executorch/backends/webgpu:webgpu_model_loader",
            "//executorch/extension/tensor:tensor",
        ],
    )
