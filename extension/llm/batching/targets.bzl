load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Mirrors the extension_llm_batching CMake targets.

    `batching` is the scheduler, the executor seam and the runner, free of
    ExecuTorch runtime types. `session_table` is what every executor over a
    multi-sequence cache shares. `module_executor` implements the seam against
    a program and a KV cache.
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
        name = "executor_utils",
        exported_headers = ["executor_utils.h"],
        header_namespace = "executorch/extension/llm/batching",
        exported_deps = [
            ":batching",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:backend_options_map",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "session_table",
        srcs = [
            "util/session_table.cpp",
        ],
        exported_headers = [
            "util/session_table.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":batching",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        deps = [
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/platform:platform",
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
            ":session_table",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/extension/llm/runner:stats",
            "//executorch/extension/module:module",
            "//executorch/runtime/core:core",
        ],
        deps = [
            ":executor_utils",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.python_library(
        name = "sampler",
        srcs = [
            "sampler.py",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//caffe2:torch",
        ],
    )
