load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "et_pal",
        srcs = ["et_pal.cpp"],
        link_whole = True,
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS"
        ],
        exported_deps = [
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.python_library(
        name = "etdump",
        srcs = ["etdump.py"],
        visibility = [
            "//executorch/backends/cadence/...",
            "@EXECUTORCH_CLIENTS"
        ],
        deps = [
            "fbcode//executorch/devtools:lib",
            "fbcode//executorch/devtools/inspector:inspector_utils",
            "fbsource//third-party/pypi/tabulate:tabulate",
        ],
    )
