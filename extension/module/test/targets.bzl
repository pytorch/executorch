load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_test(
            name = "test" + aten_suffix,
            srcs = [
                "module_test.cpp",
            ],
            deps = [
                "//executorch/kernels/portable:generated_lib" + aten_suffix,
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/module:module" + aten_suffix,
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
            env = {
                "RESOURCES_PATH": "$(location :resources)/resources",
            },
            platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
            compiler_flags = [
                "-Wno-error=deprecated-declarations",
            ],
        )

    runtime.filegroup(
        name = "resources",
        srcs = native.glob([
            "resources/**",
        ]),
    )
