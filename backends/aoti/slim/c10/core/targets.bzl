load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define targets for SlimTensor c10 core module."""

    # Header-only library for DeviceType
    runtime.cxx_library(
        name = "device_type",
        headers = [
            "DeviceType.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/platform:platform",
        ],
    )

    # Header-only library for Device
    runtime.cxx_library(
        name = "device",
        headers = [
            "Device.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":device_type",
            "//executorch/runtime/platform:platform",
        ],
    )

    # Header-only library for ScalarType
    runtime.cxx_library(
        name = "scalar_type",
        headers = [
            "ScalarType.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/core/portable_type:portable_type",
            "//executorch/runtime/platform:platform",
        ],
    )

    # Header-only library for SizesAndStrides
    runtime.cxx_library(
        name = "sizes_and_strides",
        headers = [
            "SizesAndStrides.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/macros:macros",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )

    # Header-only library for Contiguity
    runtime.cxx_library(
        name = "contiguity",
        headers = [
            "Contiguity.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    # Header-only library for WrapDimMinimal
    runtime.cxx_library(
        name = "wrap_dim_minimal",
        headers = [
            "WrapDimMinimal.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10/macros:macros",
            "//executorch/runtime/platform:platform",
        ],
    )

    # Combined c10 core library
    runtime.cxx_library(
        name = "core",
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":contiguity",
            ":device",
            ":device_type",
            ":scalar_type",
            ":sizes_and_strides",
            ":wrap_dim_minimal",
        ],
    )
