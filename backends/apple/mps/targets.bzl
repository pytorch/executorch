#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_xplat = False, platforms = []):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    kwargs = {
        "compiler_flags": [
            "-DEXIR_MPS_DELEGATE=1",
            "-Wno-global-constructors",
            "-Wno-missing-prototypes",
            "-Wno-nullable-to-nonnull-conversion",
            "-Wno-unused-const-variable",
            "-fno-objc-arc",
        ],
        "deps": [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        "exported_deps": [
            "//executorch/runtime/backend:interface",
        ],
        "headers": [
            "runtime/MPSCompiler.h",
            "runtime/MPSDevice.h",
            "runtime/MPSExecutor.h",
            "runtime/MPSStream.h",
            "utils/MPSGraphPackageExport.h",
            "utils/OperationUtils.h",
        ],
        "name": "mps",
        "srcs": [
            "runtime/MPSBackend.mm",
            "runtime/MPSCompiler.mm",
            "runtime/MPSDevice.mm",
            "runtime/MPSExecutor.mm",
            "runtime/MPSStream.mm",
        ],
        "visibility": [
            "//executorch/backends/apple/...",
            "//executorch/examples/...",
            "//executorch/exir/backend:backend_lib",
            "//executorch/extension/pybindings/...",
            "//executorch/runtime/backend/...",
            "//executorch/sdk/runners/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    }

    if is_xplat:
        kwargs["fbobjc_frameworks"] = [
            "Metal",
            "MetalPerformanceShaders",
            "MetalPerformanceShadersGraph",
        ]
        kwargs["fbobjc_ios_target_sdk_version"] = "17.0"
        kwargs["platforms"] = platforms

    if runtime.is_oss or is_xplat:
        runtime.cxx_library(**kwargs)
