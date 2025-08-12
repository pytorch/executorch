load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets():
    python_unittest(
        name = "test_llm_config",
        srcs = [
            "test_llm_config.py",
        ],
        deps = [
            "//executorch/extension/llm/export/config:llm_config",
        ],
    )
