load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Debug-dump formatting shared by every renderer of the IR (header-only).
    runtime.cxx_library(
        name = "format",
        exported_headers = [
            "Format.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
