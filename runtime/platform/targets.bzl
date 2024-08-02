load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(":log.bzl", "get_et_logging_flags")

def _select_pal(dict_):
    """Returns an element of `dict_` based on the value of the
    `executorch.pal_default` build config value. Fails if no corresponding entry
    exists.
    """
    pal_default = native.read_config("executorch", "pal_default", "posix")
    if not pal_default in dict_:
        fail("Missing key for executorch.pal_default value '{}' in dict '{}'".format(pal_default, dict_))
    return dict_[pal_default]

def profiling_enabled():
    return native.read_config("executorch", "prof_enabled", "false") == "true"

def get_profiling_flags():
    profiling_flags = []
    if profiling_enabled():
        profiling_flags += ["-DPROFILING_ENABLED"]
    prof_buf_size = native.read_config("executorch", "prof_buf_size", None)
    if prof_buf_size != None:
        if not profiling_enabled():
            fail("Cannot set profiling buffer size without enabling profiling first.")
        profiling_flags += ["-DMAX_PROFILE_EVENTS={}".format(prof_buf_size), "-DMAX_MEM_PROFILE_EVENTS={}".format(prof_buf_size)]
    num_prof_blocks = native.read_config("executorch", "num_prof_blocks", None)
    if num_prof_blocks != None:
        if not profiling_enabled():
            fail("Cannot configure number of profiling blocks without enabling profiling first.")
        profiling_flags += ["-DMAX_PROFILE_BLOCKS={}".format(num_prof_blocks)]
    return profiling_flags

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Default implementations of pal functions. These are weak symbols, so
    # client defined implementations will overide them.
    runtime.cxx_library(
        name = "platform_private",
        srcs = _select_pal({
            "minimal": ["default/minimal.cpp"],
            "posix": ["default/posix.cpp"],
        }),
        deps = [
            ":pal_interface",
        ],
        visibility = [
            "//executorch/core/...",
        ],
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading .so library we don't want
        # it to depend on other .so libraries because that way we have to
        # specify library directory path.
        force_static = True,
    )

    # Interfaces for executorch users
    runtime.cxx_library(
        name = "platform",
        exported_headers = [
            "abort.h",
            "assert.h",
            "clock.h",
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
        exported_preprocessor_flags = get_profiling_flags() + get_et_logging_flags(),
        exported_deps = [
            "//executorch/runtime/platform:pal_interface",
            ":compiler",
            ":platform_private",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading so library we don't want
        # it to depend on other so libraries because that way we have to
        # specify library directory path.
        force_static = True,
    )

    # Library for backend implementers to define implementations against.
    runtime.cxx_library(
        name = "pal_interface",
        exported_headers = [
            "platform.h",
            "system.h",
            "types.h",
        ],
        exported_deps = [
            ":compiler",
        ],
        exported_preprocessor_flags = select(
            {
                "DEFAULT": [],
                "ovr_config//os:linux": ["-DET_USE_LIBDL"],
                "ovr_config//os:macos": ["-DET_USE_LIBDL"],
            },
        ),
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
