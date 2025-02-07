load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

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
