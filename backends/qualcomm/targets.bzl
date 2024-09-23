load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


# Construct the input and output file names. All input and output files rely on scalar_type file.
SCHEMA_NAME = "schema"

INPUT_SCHEMA = "serialization/" + SCHEMA_NAME + ".fbs"

OUTPUT_SCHEMA_HEADER = SCHEMA_NAME + "_generated.h"

SCHEMA_GEN_RULE_NAME = "schema_generated"

SCHEMA_LIRRARY_NAME = SCHEMA_NAME

def generate_schema_header(rule_name, srcs, headers, default_header):
    """Generate header file given flatbuffer schema
    """
    runtime.genrule(
        name = rule_name,
        srcs = srcs,
        # We're only generating a single file, so it seems like we could use
        # `out`, but `flatc` takes a directory as a parameter, not a single
        # file. Use `outs` so that `${OUT}` is expanded as the containing
        # directory instead of the file itself.
        outs = {header: [header] for header in headers},
        default_outs = [default_header],
        cmd = " ".join([
            "$(exe {})".format(runtime.external_dep_location("flatc")),
            "--cpp",
            "--cpp-std c++11",
            "--gen-mutable",
            "--scoped-enums",
            "-o ${OUT}",
            "${SRCS}",
            # Let our infra know that the file was generated.
            " ".join(["&& echo // @" + "generated >> ${OUT}/" + header for header in headers]),
        ]),
        visibility = [],  # Private
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    generate_schema_header(
        SCHEMA_GEN_RULE_NAME,
        [INPUT_SCHEMA],
        [OUTPUT_SCHEMA_HEADER],
        OUTPUT_SCHEMA_HEADER,
    )
    # Header-only library target with the generate executorch program schema header.
    runtime.cxx_library(
        name = "schema",
        srcs = [],
        visibility = [
            # Lock this down as tightly as possible to ensure that flatbuffers
            # are an implementation detail. Ideally this list would only include
            # //executorch/runtime/executor/...
            "//executorch/codegen/tools/...",
            "//executorch/runtime/executor/...",
            "//executorch/backends/qualcomm/...",
            "//executorch/backends/qualcomm/runtime/...",
        ],
        exported_headers = {
            OUTPUT_SCHEMA_HEADER: ":{}[{}]".format(SCHEMA_GEN_RULE_NAME, OUTPUT_SCHEMA_HEADER),
        },
        exported_external_deps = ["flatbuffers-api"],
        define_static_target = True,
        platforms = [ANDROID],
    )


    runtime.cxx_library(
        name = "qnn_executorch_backend",
        srcs = [],
        headers = [],
        define_static_target = True,
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "fbsource//third-party/qualcomm/qnn:api",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/backends/qualcomm/runtime:runtime",
        ],
        exported_deps = [
            ":schema",
        ],
    )
