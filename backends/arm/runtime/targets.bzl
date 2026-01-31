load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "vela_bin_stream",
        srcs = ["VelaBinStream.cpp"],
        exported_headers = ["VelaBinStream.h"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/core:core",
        ],
    )
    runtime.cxx_library(
        name = "arm_backend",
        srcs = ["EthosUBackend.cpp"],
        headers = [],
        compatible_with = ["ovr_config//cpu:arm32-embedded", "ovr_config//cpu:arm32-embedded-fpu"],
        # arm_executor_runner.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            ":vela_bin_stream",
            "//executorch/runtime/core:core",
            "fbsource//third-party/ethos-u-core-driver:core_driver",
        ],
    )
