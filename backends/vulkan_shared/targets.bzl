load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    """Defines the shared Vulkan runtime and its unit tests."""

    # The device has no ExecuTorch runtime deps, so the Vulkan compute API can
    # export it without pulling the runtime into binaries that only use the
    # compute API.
    runtime.cxx_library(
        name = "device",
        srcs = ["runtime/SharedVulkanDevice.cpp"],
        exported_headers = [
            "runtime/SharedVulkanDevice.h",
            "runtime/export.h",
        ],
        force_static = False,
        preprocessor_flags = [
            "-DEXECUTORCH_VULKAN_SHARED_BUILDING",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "fbsource//third-party/khronos:vulkan-headers",
        ],
    )

    # This target must be shareable rather than force-static. VGF and Vulkan can
    # live in separate DSOs, but both must resolve the same process registry.
    runtime.cxx_library(
        name = "runtime",
        srcs = [
            "runtime/SharedVulkanDeviceRegistry.cpp",
            "runtime/SharedVulkanRuntimeConfig.cpp",
        ],
        exported_headers = [
            "runtime/SharedVulkanDeviceRegistry.h",
            "runtime/SharedVulkanRuntimeConfig.h",
        ],
        force_static = False,
        preprocessor_flags = [
            "-DEXECUTORCH_VULKAN_SHARED_BUILDING",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/backends/vulkan_shared:device",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_test(
        name = "shared_vulkan_runtime_config_test",
        srcs = ["runtime/test/SharedVulkanRuntimeConfigTest.cpp"],
        deps = [
            ":runtime",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_test(
        name = "shared_gpu_device_registry_test",
        srcs = ["runtime/test/SharedVulkanDeviceRegistryTest.cpp"],
        deps = [
            ":runtime",
            "//executorch/runtime/core:core",
        ],
    )
