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
        "name": "mps",
        "compiler_flags": [
            "-DEXIR_MPS_DELEGATE=1",
            "-Wno-global-constructors",
            "-Wno-missing-prototypes",
            "-Wno-nullable-to-nonnull-conversion",
            "-Wno-undeclared-selector",
            "-Wno-unused-const-variable",
            "-Wno-unused-variable",
            "-fno-objc-arc",
            "-std=c++17",
        ],
        "deps": [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":mps_schema",
        ],
        "exported_deps": [
            "//executorch/runtime/backend:interface",
            ":mps_schema",
        ],
        "headers": native.glob([
            "runtime/*.h",
            "runtime/operations/*.h",
        ]),
        "srcs": native.glob([
            "runtime/*.mm",
            "runtime/operations/*.mm",
        ]),
        "visibility": [
            "//executorch/backends/apple/...",
            "//executorch/examples/...",
            "//executorch/exir/backend:backend_lib",
            "//executorch/extension/pybindings/...",
            "//executorch/runtime/backend/...",
            "//executorch/devtools/runners/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        "link_whole": True,
    }

    if is_xplat:
        kwargs["fbobjc_frameworks"] = [
            "Foundation",
            "Metal",
            "MetalPerformanceShaders",
            "MetalPerformanceShadersGraph",
        ]
        kwargs["platforms"] = platforms

    if runtime.is_oss or is_xplat:
        runtime.genrule(
            name = "gen_mps_schema",
            srcs = [
                "serialization/schema.fbs",
            ],
            outs = {
                "schema_generated.h": ["schema_generated.h"],
            },
            cmd = " ".join([
                "$(exe {})".format(runtime.external_dep_location("flatc")),
                "--cpp",
                "--cpp-std c++11",
                "--scoped-enums",
                "-o ${OUT}",
                "${SRCS}",
            ]),
            default_outs = ["."],
        )

        runtime.cxx_library(
            name = "mps_schema",
            srcs = [],
            exported_headers = {
                "schema_generated.h": ":gen_mps_schema[schema_generated.h]",
            },
            exported_external_deps = ["flatbuffers-api"],
            visibility = [
                "//executorch/backends/apple/...",
                "//executorch/examples/...",
            ],
        )

        runtime.cxx_library(**kwargs)
