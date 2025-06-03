load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

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

    runtime.python_library(
        name = "llm_config_utils",
        srcs = [
            "llm_config_utils.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.llama.config",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            ":llm_config",
        ],
    )
