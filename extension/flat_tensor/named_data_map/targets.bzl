load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "data_map",
        srcs = [
            "data_map.cpp",
        ],
        exported_headers = ["data_map.h"],
        deps = [
            "//executorch/extension/flat_tensor/serialize:schema",
            "//executorch/extension/flat_tensor/serialize:serialize",
            "//executorch/extension/flat_tensor/serialize:generated_headers",
            "//executorch/extension/flat_tensor/serialize:flat_tensor_header",
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
