load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "state_dict_util",
        srcs = [
            "state_dict_util.cpp",
        ],
        exported_headers = [
            "state_dict_util.h",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:named_data_map",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core:core",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "training_module" + aten_suffix,
            srcs = [
                "training_module.cpp",
            ],
            exported_headers = [
                "training_module.h",
            ],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
            ],
        )
