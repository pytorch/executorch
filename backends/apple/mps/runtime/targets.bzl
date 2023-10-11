#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    if runtime.is_oss:
        runtime.cxx_library(
            name = "MPSBackend",
            srcs = [
              "MPSExecutor.mm",
              "MPSCompiler.mm",
              "MPSBackend.mm",
              "MPSStream.mm",
              "MPSDevice.mm",
            ] + native.glob(["utils/OperationUtils.mm"]),
            # lang_preprocessor_flags = 'objective-c++'
            visibility = [
                "//executorch/exir/backend:backend_lib",
                "//executorch/backends/apple/...",
                "//executorch/runtime/backend/...",
                "//executorch/extension/pybindings/...",
                "//executorch/sdk/runners/...",
                "//executorch/test/...",
                "//executorch/examples/...",
                "@EXECUTORCH_CLIENTS",
            ],
            headers = native.glob([
              "runtime/*.h",
            ]),
            # registration of backends is done through a static global
            compiler_flags = ["-Wno-global-constructors"],
            external_deps = [
            "gflags",
            ],
            exported_deps = [
                "//executorch/runtime/backend:interface",
            ],
            link_whole = True,
        )
