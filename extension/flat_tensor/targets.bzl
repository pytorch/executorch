load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "flat_tensor_data_map",
        srcs = [
            "flat_tensor_data_map.cpp",
        ],
        exported_headers = ["flat_tensor_data_map.h"],
        exported_deps = [
            "//executorch/extension/flat_tensor/serialize:flat_tensor_header",
        ],
        deps = [
            "//executorch/extension/flat_tensor/serialize:generated_headers",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/core:named_data_map",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        visibility = [
            "//executorch/...",
        ],
    )
