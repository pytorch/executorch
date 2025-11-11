load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/target_determinator/macros:ci.bzl", "ci")

def define_common_targets():
    runtime.cxx_library(
        name = "nxp_backend_base",
        srcs = ["NeutronBackend.cpp"],
        exported_headers = [
            "NeutronDriver.h",
            "NeutronErrors.h",
        ],
        link_whole = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors", "-fno-rtti", "-DNO_HEAP_USAGE"],
        labels = [ci.skip_target()],
        visibility = [
            "//executorch/backends/nxp/runtime/fb:nxp_fb_backend",
            "//executorch/backends/nxp/runtime/fb:nxp_hifi_fb_backend",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )
