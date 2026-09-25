load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    """Defines the shared Vulkan runtime and its unit tests."""

    # This target must be shareable rather than force-static. VGF and Vulkan can
    # live in separate DSOs, but both must resolve the same process registry.
    runtime.cxx_library(
        name = "runtime",
        srcs = [
            "runtime/SharedVulkanContext.cpp",
            "runtime/SharedVulkanContextRegistry.cpp",
            "runtime/SharedVulkanRuntimeConfig.cpp",
        ],
        exported_headers = [
            "runtime/SharedVulkanContext.h",
            "runtime/SharedVulkanContextRegistry.h",
            "runtime/SharedVulkanRuntimeConfig.h",
            "runtime/export.h",
        ],
        force_static = False,
        preprocessor_flags = [
            "-DEXECUTORCH_VULKAN_SHARED_BUILDING",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "fbsource//third-party/khronos:vulkan-headers",
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
        name = "shared_gpu_context_registry_test",
        srcs = ["runtime/test/SharedVulkanContextRegistryTest.cpp"],
        deps = [
            ":runtime",
            "//executorch/runtime/core:core",
        ],
    )
