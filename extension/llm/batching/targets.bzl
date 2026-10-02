load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Mirrors the extension_llm_batching CMake targets.

    `batching` is the scheduler, the executor seam and the runner, free of
    ExecuTorch runtime types. `module_executor` implements the seam against a
    program and a KV cache.
    """
    runtime.cxx_library(
        name = "batching",
        srcs = [
            "metrics.cpp",
            "runner.cpp",
        ],
        exported_headers = [
            "decode_first_scheduler.h",
            "executor.h",
            "metrics.h",
            "prefix_cache.h",
            "runner.h",
            "scheduler.h",
            "types.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/platform:compiler",
        ],
    )

    runtime.cxx_library(
        name = "module_executor",
        srcs = [
            "module_executor.cpp",
        ],
        exported_headers = [
            "module_executor.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":batching",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/extension/llm/runner:stats",
            "//executorch/extension/module:module",
            "//executorch/runtime/core:core",
        ],
        deps = [
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/platform:platform",
        ],
    )
