load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Mirrors extension_llm_batching_test in CMakeLists.txt."""
    runtime.cxx_test(
        name = "test",
        srcs = [
            "prefix_cache_test.cpp",
            "runner_test.cpp",
            "scheduler_test.cpp",
        ],
        headers = [
            "fake_executor.h",
        ],
        deps = [
            "//executorch/extension/llm/batching:batching",
        ],
    )
