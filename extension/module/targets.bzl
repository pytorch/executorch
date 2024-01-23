load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_default_executorch_platforms", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "module" + aten_suffix,
            srcs = [
                "module.cpp",
            ],
            exported_headers = [
                "module.h",
            ],
            platforms = ["Default"] if aten_mode else get_default_executorch_platforms(),
            define_static_target = not aten_mode,
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/extension/memory_allocator:malloc_memory_allocator",
                "//executorch/extension/data_loader:mmap_data_loader",
            ],
            exported_deps = [
                "//executorch/runtime/executor:program" + aten_suffix,
            ],
        )
