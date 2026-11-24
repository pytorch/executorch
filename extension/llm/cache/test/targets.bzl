load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.cxx_test(
        name = "cache_test",
        srcs = ["cache_test.cpp"],
        deps = [
            "//executorch/extension/llm/cache:kv_cache",
            "//executorch/test/utils:utils",
        ],
    )
