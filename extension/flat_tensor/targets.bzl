load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "flat_tensor_data_map" + aten_suffix,
            srcs = [
                "flat_tensor_data_map.cpp",
            ],
            exported_headers = ["flat_tensor_data_map.h"],
            deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:evalue",
                "//executorch/runtime/core:named_data_map" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util",
            ],
            exported_deps = [
                "//executorch/extension/flat_tensor/serialize:flat_tensor_header",
                "//executorch/extension/flat_tensor/serialize:generated_headers",
            ],
            visibility = ["PUBLIC"],
        )
