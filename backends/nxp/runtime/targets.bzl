load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "nxp_backend",
        srcs = ["NeutronBackend.cpp"],
        headers = ["NeutronDriver.h", "NeutronErrors.h"],
        compatible_with = ["ovr_config//cpu:arm32-embedded", "@fbsource//arvr/firmware/projects/smartglasses/config:embedded-mcu-rtos"],
        # Neutron runtime needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors", "-fno-rtti", "-DNO_HEAP_USAGE"],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "fbsource//arvr/third-party/toolchains/nxp-sdk/2.16.0/middleware/eiq/executorch/third-party/neutron/rt700:libNeutron",
        ],
    )
