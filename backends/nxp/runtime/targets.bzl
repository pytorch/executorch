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
        # NEUTRON_NPU_POWER_GATING clock-gates the NPU around each inference
        # (neutronResume/neutronSuspend in NeutronBackend.cpp).
        compiler_flags = ["-Wno-global-constructors", "-fno-rtti", "-DNO_HEAP_USAGE", "-DNEUTRON_NPU_POWER_GATING"],
        labels = [ci.skip_target()],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )
