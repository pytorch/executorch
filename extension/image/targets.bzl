load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "image_processor" + aten_suffix,
            srcs = [
                "image_processor_common.cpp",
                "image_processor.cpp",
            ],
            exported_headers = [
                "image_processor.h",
                "image_processor_config.h",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/runtime/core:core",
            ],
            external_deps = [
                "stb",
            ],
        )
