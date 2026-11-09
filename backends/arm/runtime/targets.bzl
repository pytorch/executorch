# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _persistent_region_flags():
    if native.read_config("executorch", "ethosu_persistent_region", "false") == "true":
        return ["-DETHOSU_PERSISTENT_REGION"]
    if runtime.is_oss:
        return []
    return select({
        "DEFAULT": [],
        "fbsource//xplat/executorch/tools/buck/constraints:ethosu-persistent-region-enabled": ["-DETHOSU_PERSISTENT_REGION"],
    })

def _vgf_zero_copy_flags():
    if native.read_config("executorch", "vgf_zero_copy_io", "false") == "true":
        return ["-DEXECUTORCH_VGF_ZERO_COPY_IO=1"]
    return ["-DEXECUTORCH_VGF_ZERO_COPY_IO=0"]

def define_common_targets():
    runtime.cxx_library(
        name = "ethosu_invocation_context",
        exported_headers = ["EthosUBackend_Pmu.h"],
        exported_deps = ["//executorch/runtime/core:event_tracer"],
        visibility = ["PUBLIC"],
    )
    runtime.cxx_library(
        name = "ethosu_execution_stats",
        srcs = ["EthosUExecutionStats.cpp"],
        compatible_with = ["ovr_config//cpu:arm32-embedded", "ovr_config//cpu:arm32-embedded-fpu"],
        exported_headers = ["EthosUExecutionStats.h"],
        exported_deps = [
            ":ethosu_invocation_context",
            "//executorch/runtime/core:core",
            "fbsource//third-party/ethos-u-core-driver:core_driver_headers_only",
        ],
        visibility = ["PUBLIC"],
    )
    runtime.cxx_library(
        name = "vela_bin_stream",
        srcs = ["VelaBinStream.cpp"],
        exported_headers = ["VelaBinStream.h"],
        preprocessor_flags = _persistent_region_flags(),
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
        deps = [
            "//executorch/runtime/core:named_data_map",
        ],
    )
    runtime.cxx_library(
        name = "arm_backend",
        srcs = [
            "EthosUBackend.cpp",
            "EthosUBackend_CoreDriver.cpp",
            "EthosUBackend_IoMemcpy.cpp",
        ],
        headers = ["EthosUBackend_Internal.h"],
        compatible_with = ["ovr_config//cpu:arm32-embedded", "ovr_config//cpu:arm32-embedded-fpu"],
        # arm_executor_runner.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        preprocessor_flags = _persistent_region_flags(),
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            ":ethosu_invocation_context",
            ":vela_bin_stream",
            "//executorch/runtime/core:core",
            "fbsource//third-party/ethos-u-core-driver:core_driver_headers_only",
        ],
    )
    runtime.cxx_test(
        name = "ethosu_planned_scratch_test",
        srcs = [
            "EthosUBackend.cpp",
            "EthosUBackend_IoMemcpy.cpp",
            "tests/ethos-u/test_ethosu_planned_scratch.cpp",
        ],
        headers = ["EthosUBackend_Internal.h"],
        compiler_flags = ["-Wno-global-constructors"],
        preprocessor_flags = _persistent_region_flags(),
        deps = [
            ":ethosu_invocation_context",
            ":vela_bin_stream",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )
    runtime.cxx_library(
        name = "vgf_backend",
        srcs = [
            "VGFBackend.cpp",
            "VGFExecutionStats.cpp",
            "VGFNeuralStatistics.cpp",
            "VGFSetup.cpp",
            "VGFZeroCopy.cpp",
            # Volk must be compiled directly into this target so its global
            # function-pointer variables live in the same linkage unit.
            # Linking from a separate static library causes the linker to
            # drop the symbols when building a shared library.
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:volk_arm_src",
        ],
        exported_headers = [
            "VGFBoundaryIO.h",
            "VGFExecutionStats.h",
            "VGFMutableAlias.h",
            "VGFNeuralStatistics.h",
            "VGFSetup.h",
            "VGFShapeUtils.h",
            "VGFVulkanFeatures.h",
            "VGFZeroCopy.h",
        ],
        platforms = CXX,
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
        preprocessor_flags = _vgf_zero_copy_flags(),
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "fbsource//third-party/pypi/ai-ml-sdk-vgf-library/0.11.0:vgf",
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:volk_arm",
            "fbsource//third-party/vulkan-headers-1.4.343/v1.4.343/src:vulkan-headers",
        ],
    )
