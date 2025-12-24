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
            "//executorch/runtime/platform:platform",
        ],
    )

    # Combined c10 core library
    runtime.cxx_library(
        name = "core",
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":device",
            ":device_type",
            ":scalar_type",
        ],
    )
