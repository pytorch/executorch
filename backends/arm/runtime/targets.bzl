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
        srcs = [
            "EthosUBackend.cpp",
            "EthosUBackend_Cortex_M.cpp",
        ],
        headers = ["EthosUBackend_Internal.h"],
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
    runtime.cxx_library(
        name = "vgf_backend",
        srcs = [
            "VGFBackend.cpp",
            "VGFSetup.cpp",
            # Volk must be compiled directly into this target so its global
            # function-pointer variables live in the same linkage unit.
            # Linking from a separate static library causes the linker to
            # drop the symbols when building a shared library.
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:volk_arm_src",
        ],
        exported_headers = ["VGFSetup.h"],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        compiler_flags = [
            "-Wno-global-constructors",
            "-Wno-header-hygiene",
            "-Wno-unused-variable",
            "-Wno-missing-field-initializers",
            "-DUSE_VULKAN_WRAPPER",
            "-DUSE_VULKAN_VOLK",
        ],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "fbsource//third-party/arm-vgf-library/v0.9.0/src:vgf",
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:volk_arm",
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:vulkan-headers",
        ],
    )
