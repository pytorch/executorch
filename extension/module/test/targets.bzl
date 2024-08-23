load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "test",
        srcs = [
            "module_test.cpp",
        ],
        deps = [
            "//executorch/kernels/portable:generated_lib",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/module:module",
        ],
        env = {
            "RESOURCES_PATH": "$(location :resources)/resources",
        },
    )

    runtime.filegroup(
        name = "resources",
        srcs = native.glob([
            "resources/**",
        ]),
    )
