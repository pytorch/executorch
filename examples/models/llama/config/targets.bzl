load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets():
    runtime.python_library(
        name = "llm_config",
        srcs = [
            "llm_config.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.llama.config",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    python_unittest(
        name = "test_llm_config",
        srcs = [
            "test_llm_config.py",
        ],
        deps = [
            ":llm_config",
        ],
    )
