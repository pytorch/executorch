load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Only for OSS
    if runtime.is_oss:
        native.cxx_library(
            name = "nlohmann-json",
            exported_deps = ["root//extension/llm/tokenizers/third-party:nlohmann_json"],
            visibility = ["PUBLIC"],
        )
