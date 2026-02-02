load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "module" + aten_suffix,
            srcs = [
                "module.cpp",
            ],
            exported_headers = [
                "module.h",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/extension/memory_allocator:malloc_memory_allocator",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/data_loader:mmap_data_loader",
                "//executorch/extension/flat_tensor:flat_tensor_data_map" + aten_suffix,
                "//executorch/extension/named_data_map:merged_data_map" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/runtime/executor:program_no_prim_ops" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "bundled_module" + aten_suffix,
            srcs = [
                "bundled_module.cpp",
            ],
            exported_headers = [
                "bundled_module.h",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/devtools/bundled_program:runtime" + aten_suffix,
                "//executorch/devtools/bundled_program/schema:bundled_program_schema_fbs",
            ],
            exported_deps = [
                "//executorch/extension/module:module" + aten_suffix,
            ],
        )
