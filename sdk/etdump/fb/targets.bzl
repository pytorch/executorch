load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

ETDUMP_STEM = "etdump_schema"
ETDUMP_SCHEMA = ETDUMP_STEM + ".fbs"
ETDUMP_GEN_RULE_NAME = "generate_etdump"
ETDUMP_LIBRARY_NAME = ETDUMP_STEM

SCALAR_TYPE_STEM = "scalar_type"
SCALAR_TYPE = SCALAR_TYPE_STEM + ".fbs"

# flatbuffers:flatc

ETDUMP_SCHEMA_HEADER = ETDUMP_STEM + "_generated.h"
OUTPUT_SCALAR_TYPE_HEADER = SCALAR_TYPE_STEM + "_generated.h"

def generate_schema_header(rule_name, srcs, headers, default_header):
    """
    Generate header files for ETDump schema
    """

    runtime.genrule(
        name = rule_name,
        srcs = srcs,
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
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.export_file(
        name = ETDUMP_SCHEMA,
        visibility = ["//executorch/..."],
    )

    generate_schema_header(
        ETDUMP_GEN_RULE_NAME,
        [ETDUMP_SCHEMA, SCALAR_TYPE],
        [ETDUMP_SCHEMA_HEADER, OUTPUT_SCALAR_TYPE_HEADER],
        ETDUMP_SCHEMA_HEADER,
    )

    runtime.cxx_library(
        name = ETDUMP_LIBRARY_NAME,
        srcs = [],
        visibility = ["//executorch/..."],
        exported_headers = {
            ETDUMP_SCHEMA_HEADER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME, ETDUMP_SCHEMA_HEADER),
            OUTPUT_SCALAR_TYPE_HEADER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME, OUTPUT_SCALAR_TYPE_HEADER),
        },
        exported_external_deps = ["flatbuffers-api"],
    )

    runtime.cxx_library(
        name = "etdump",
        srcs = ["etdump.cpp"],
        exported_headers = ["etdump.h"],
        deps = [
            ":etdump_gen",
            "//executorch/runtime/core:core",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "etdump_gen",
        srcs = ["etdump_gen.cpp"],
        exported_headers = ["etdump_gen.h"],
        deps = [],
        exported_deps = [
            ":etdump_schema",
            "//executorch/runtime/platform:platform",
            "//executorch/runtime/core:memory_allocator",
        ],
        visibility = [
            "//executorch/...",
        ],
    )
