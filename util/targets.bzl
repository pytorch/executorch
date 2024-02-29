load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "read_file",
        srcs = ["read_file.cpp"],
        exported_headers = ["read_file.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:compiler",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        # DEPRECATED: Remove this once all users have migrated to
        # extension/runner_util:inputs.
        runtime.cxx_library(
            name = "util" + aten_suffix,
            srcs = [],
            exported_headers = ["util.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/extension/runner_util:inputs" + aten_suffix,
                "//executorch/runtime/core:core",
                "//executorch/runtime/executor:program" + aten_suffix,
            ],
        )

    if not runtime.is_oss:
        runtime.python_library(
            name = "python_profiler",
            srcs = [
                "python_profiler.py",
            ],
            deps = [
                "fbsource//third-party/pypi/snakeviz:snakeviz",
                "fbsource//third-party/pypi/tornado:tornado",
            ],
            visibility = ["@EXECUTORCH_CLIENTS"],
            _is_external_target = True,
        )
