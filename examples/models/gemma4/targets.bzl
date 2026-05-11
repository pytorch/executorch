load("@fbsource//tools/build_defs:platform_defs.bzl", "APPLE", "IOS", "MACOSX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

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
        preprocessor_flags = [
            "-DENABLE_XNNPACK_SHARED_WORKSPACE",
        ],
        deps = _KERNEL_BACKEND_DEPS + [
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/llm/runner:runner_lib",
            "//pytorch/tokenizers:headers",
        ],
    )

    _ANE_MACOS_COMPILER_FLAGS = [
        "-g0",
        "-Oz",
        "-fexceptions",
        "-frtti",
        "-Wno-deprecated-declarations",
        "-Wno-global-constructors",
        "-Wno-error",
        "-Wno-nonportable-include-path",
    ]

    runtime.cxx_library(
        name = "ane_text_main_lib",
        srcs = ["ane_text_runner.cpp"],
        apple_sdks = (IOS, MACOSX),
        compiler_flags = _ANE_MACOS_COMPILER_FLAGS,
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/core:core",
            "//pytorch/tokenizers:headers",
        ] + select({
            "DEFAULT": [],
            "ovr_config//os:iphoneos": [
                "//executorch/backends/apple/coreml:coreml",
            ],
        }),
        exported_deps = [] + select({
            "DEFAULT": [],
            "ovr_config//os:iphoneos": [
                "//executorch/backends/apple/coreml:coreml",
            ],
        }),
        external_deps = [
            "gflags",
        ],
        preprocessor_flags = ["-DET_USE_THREADPOOL", "-DENABLE_XNNPACK_SHARED_WORKSPACE"],
    )

    runtime.cxx_binary(
        name = "ane_text_main",
        deps = [
            ":ane_text_main_lib",
        ] + _KERNEL_BACKEND_DEPS + GEN_KERNEL_BACKEND_DEPS + _get_torchao_lowbit_deps() + select({
            "DEFAULT": [],
            "ovr_config//os:macos": [
                "//executorch/backends/apple/coreml:coreml",
            ],
        }),
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "ane_text_main_ios_lib",
        apple_sdks = (IOS,),
        platforms = (APPLE,),
        visibility = ["PUBLIC"],
        deps = [
            ":ane_text_main_lib",
            "//xplat/cria/benchmark:ios_benchmark_main_lib",
        ],
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
        preprocessor_flags = ["-DET_USE_THREADPOOL", "-DENABLE_XNNPACK_SHARED_WORKSPACE"],
    )
