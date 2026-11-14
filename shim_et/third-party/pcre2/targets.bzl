load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    if runtime.is_oss:
        native.cxx_library(
            name = "pcre2-8",
            exported_deps = ["root//extension/llm/tokenizers/third-party:pcre2"],
            visibility = ["PUBLIC"],
        )
