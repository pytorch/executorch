load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.cxx_library(
        name = "step_plan",
        exported_headers = [
            "step_plan.h",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "cuda_executor",
        srcs = [
            "cuda_executor.cpp",
        ],
        exported_headers = [
            "cuda_executor.h",
        ],
        preprocessor_flags = [
            "-DCUDA_AVAILABLE=1",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":step_plan",
            "//executorch/backends/cuda/runtime:cuda_backend",
            "//executorch/extension/llm/batching:batching",
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/extension/module:module",
        ],
        deps = [
            "//executorch/extension/llm/runner:stats",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/platform:platform",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )

    cpp_unittest(
        name = "test_step_plan",
        srcs = [
            "test/test_step_plan.cpp",
        ],
        deps = [
            ":step_plan",
        ],
    )

    # Writes the toy decoder test_cuda_executor runs. Needs a GPU: AOTI
    # compiles and autotunes on it. test_cuda_executor itself builds with CMake
    # only (see CMakeLists.txt): it runs the compiled AOTI library, which
    # resolves its shims against the host binary.
    runtime.python_binary(
        name = "export_toy_decoder",
        srcs = [
            "test/export_toy_decoder.py",
        ],
        main_function = "executorch.backends.cuda.batching.test.export_toy_decoder.main",
        keep_gpu_sections = True,
        deps = [
            "//caffe2:torch",
            "//executorch/backends/cuda:cuda_backend",
            "//executorch/backends/cuda:cuda_partitioner",
            "//executorch/backends/cuda:cuda_passes",
            "//executorch/exir:lib",
            "//executorch/exir/backend:compile_spec_schema",
            "//executorch/exir/passes:lib",
            "//executorch/extension/llm/cache:cache",
            "//executorch/extension/llm/export:model_metadata",
        ],
    )
