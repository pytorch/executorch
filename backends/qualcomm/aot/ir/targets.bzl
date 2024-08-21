load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm:targets.bzl", "generate_schema_header")

QCIR_NAME = "qcir"
INPUT_QCIR = QCIR_NAME + ".fbs"
OUTPUT_QCIR_HEADER = QCIR_NAME + "_generated.h"
QCIR_GEN_RULE_NAME = "qcir_generated"

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    generate_schema_header(
        QCIR_GEN_RULE_NAME,
        [INPUT_QCIR],
        [OUTPUT_QCIR_HEADER],
        OUTPUT_QCIR_HEADER,
    )

    # Header-only library target with the generate executorch program schema header.
    runtime.cxx_library(
        name = "qcir_schema",
        srcs = [],
        exported_headers = {
            OUTPUT_QCIR_HEADER: ":{}[{}]".format(QCIR_GEN_RULE_NAME, OUTPUT_QCIR_HEADER),
        },
        visibility = [
            # Lock this down as tightly as possible to ensure that flatbuffers
            # are an implementation detail. Ideally this list would only include
            # //executorch/runtime/executor/...
            "//executorch/backends/qualcomm/...",
            "//executorch/backends/qualcomm/aot/ir/...",
        ],
        exported_external_deps = ["flatbuffers-api"],
        define_static_target = True,
        platforms = [ANDROID],
    )


    runtime.cxx_library(
        name = "qcir_utils",
        srcs = [
            "qcir_utils.cpp",
        ],
        exported_headers = [
            "qcir_utils.h",
        ],
        define_static_target = True,
        platforms = [ANDROID],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "fbsource//third-party/qualcomm/qnn:api",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/backends/qualcomm/aot/wrappers:wrappers",
        ],
        exported_deps = [
            ":qcir_schema",
        ],
    )
