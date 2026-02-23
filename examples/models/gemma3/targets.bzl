load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_binary(
        name = "vision_llm_runner",
        srcs = [
            "e2e_runner.cpp",
        ],
        compiler_flags = ["-Wno-global-constructors"],
        preprocessor_flags = ["-DET_USE_THREADPOOL"],
        deps = [
            "//executorch/extension/llm/runner:multimodal_runner_lib",
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
            "//pytorch/tokenizers:tiktoken",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/configurations:optimized_native_cpu_ops",
            "//executorch/kernels/quantized:generated_lib",
            "//executorch/extension/llm/custom_ops:custom_ops",
        ],
        external_deps = [
            "gflags",
            "stb",
        ],
    )
