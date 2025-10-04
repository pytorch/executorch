load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "merged_data_map" + aten_suffix,
            srcs = [
                "merged_data_map.cpp",
            ],
            exported_headers = [
                "merged_data_map.h",
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/core:named_data_map",
                "//executorch/runtime/core:core",
            ],
        )
