load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    if runtime.is_oss:
        native.cxx_library(
            name = "re2",
            exported_deps = ["root//extension/llm/tokenizers/third-party:re2"],
            visibility = ["PUBLIC"],
        )
