load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _select_pal(dict_):
    """Returns an element of `dict_` based on the value of the
    `executorch.pal_default` build config value. Fails if no corresponding entry
    exists.
    """
    pal_default = native.read_config("executorch", "pal_default", "posix")
    if not pal_default in dict_:
        fail("Missing key for executorch.pal_default value '{}' in dict '{}'".format(pal_default, dict_))
    return dict_[pal_default]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "platform_private",
        srcs = _select_pal({
            "minimal": ["target/Minimal.cpp"],
            "posix": ["target/Posix.cpp"],
        }),
        deps = [
            ":platform",
        ],
        visibility = [
            "//executorch/core/...",
        ],
    )

    runtime.cxx_library(
        name = "platform",
        exported_headers = [
            "Platform.h",
            "System.h",
            "Types.h",
        ],
        exported_deps = [
            "//executorch/compiler:compiler",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
