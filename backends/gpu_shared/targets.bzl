load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    """Defines the shared GPU runtime and its unit tests."""

    # This target must be shareable rather than force-static. VGF and Vulkan can
    # live in separate DSOs, but both must resolve the same process registry.
    runtime.cxx_library(
        name = "runtime",
        srcs = [
            "runtime/SharedGpuContext.cpp",
            "runtime/SharedGpuContextRegistry.cpp",
            "runtime/SharedGpuRuntimeConfig.cpp",
        ],
        exported_headers = [
            "runtime/SharedGpuContext.h",
            "runtime/SharedGpuContextRegistry.h",
            "runtime/SharedGpuRuntimeConfig.h",
            "runtime/export.h",
        ],
        force_static = False,
        preprocessor_flags = [
            "-DEXECUTORCH_GPU_SHARED_BUILDING",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "fbsource//third-party/khronos:vulkan-headers",
        ],
    )

    runtime.cxx_test(
        name = "shared_gpu_runtime_config_test",
        srcs = ["runtime/test/SharedGpuRuntimeConfigTest.cpp"],
        deps = [
            ":runtime",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_test(
        name = "shared_gpu_context_registry_test",
        srcs = ["runtime/test/SharedGpuContextRegistryTest.cpp"],
        deps = [
            ":runtime",
            "//executorch/runtime/core:core",
        ],
    )
