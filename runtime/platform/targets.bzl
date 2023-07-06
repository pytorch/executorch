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

    # Base implementations of pal functions. These are weak symbols, so client
    # defined implementations will overide them.
    runtime.cxx_library(
        name = "platform_private",
        srcs = _select_pal({
            "minimal": ["target/Minimal.cpp"],
            "posix": ["target/Posix.cpp"],
        }),
        deps = [
            ":pal_interface",
        ],
        visibility = [
            "//executorch/core/...",
        ],
    )

    # Interfaces for executorch users
    runtime.cxx_library(
        name = "platform",
        exported_headers = [
            "abort.h",
            "assert.h",
            "log.h",
            "profiler.h",
            "runtime.h",
        ],
        srcs = [
            "abort.cpp",
            "log.cpp",
            "profiler.cpp",
            "runtime.cpp",
        ],
        exported_deps = [
            "//executorch/runtime/platform:pal_interface",
            ":compiler",
            ":platform_private",
            "//executorch/profiler:profiler",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Library for backend implementers to define implementations against.
    runtime.cxx_library(
        name = "pal_interface",
        exported_headers = [
            "platform.h",
            "system.h",
            "types.h",
            "hooks.h",
        ],
        exported_deps = [
            ":compiler",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Common compiler directives such as 'unlikely' or 'deprecated'
    runtime.cxx_library(
        name = "compiler",
        exported_headers = [
            "compiler.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
