load("@fbcode_macros//build_defs:export_files.bzl", "export_file")
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

# flatcc
ETDUMP_STEM_FLATCC = "etdump_schema_flatcc"
ETDUMP_SCHEMA_FLATCC = ETDUMP_STEM_FLATCC + ".fbs"
ETDUMP_GEN_RULE_NAME_FLATCC = "generate_etdump" + "_flatcc"

ETDUMP_SCHEMA_FLATCC_BUILDER = ETDUMP_STEM_FLATCC + "_builder.h"
ETDUMP_SCHEMA_FLATCC_READER = ETDUMP_STEM_FLATCC + "_reader.h"
ETDUMP_SCHEMA_FLATCC_VERIFIER = ETDUMP_STEM_FLATCC + "_verifier.h"

SCALAR_TYPE_BUILDER = SCALAR_TYPE_STEM + "_builder.h"
SCALAR_TYPE_READER = SCALAR_TYPE_STEM + "_reader.h"
SCALAR_TYPE_VERIFIER = SCALAR_TYPE_STEM + "_verifier.h"

FLATBUFFERS_COMMON_STEM = "flatbuffers_common"
FLATBUFFERS_COMMON_BUILDER = FLATBUFFERS_COMMON_STEM + "_builder.h"
FLATBUFFERS_COMMON_READER = FLATBUFFERS_COMMON_STEM + "_reader.h"

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
            "$(exe fbsource//third-party/flatbuffers:flatc)",
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

def generate_schema_header_flatcc(rule_name, srcs, headers, default_headers):
    """
    Generate header files for ETDump schema
    """
    runtime.genrule(
        name = rule_name,
        srcs = srcs,
        outs = {header: [header] for header in headers},
        default_outs = default_headers,
        cmd = " ".join([
            "$(exe fbsource//arvr/third-party/flatcc:flatcc-cli)",
            "-cwr",
            "-o ${OUT}",
            "${SRCS}",
            # Let our infra know that the file was generated.
            " ".join(["&& echo '// @''generated' >> ${OUT}/" + header for header in headers]),
        ]),
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    export_file(
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

    export_file(
        name = ETDUMP_SCHEMA_FLATCC,
        visibility = ["//executorch/..."],
    )

    generate_schema_header_flatcc(
        ETDUMP_GEN_RULE_NAME_FLATCC,
        [ETDUMP_SCHEMA_FLATCC, SCALAR_TYPE],
        [
            ETDUMP_SCHEMA_FLATCC_BUILDER,
            ETDUMP_SCHEMA_FLATCC_READER,
            ETDUMP_SCHEMA_FLATCC_VERIFIER,
            SCALAR_TYPE_BUILDER,
            SCALAR_TYPE_READER,
            SCALAR_TYPE_VERIFIER,
            FLATBUFFERS_COMMON_BUILDER,
            FLATBUFFERS_COMMON_READER,
        ],
        [
            ETDUMP_SCHEMA_FLATCC_BUILDER,
            ETDUMP_SCHEMA_FLATCC_READER,
            ETDUMP_SCHEMA_FLATCC_VERIFIER,
        ],
    )

    runtime.cxx_library(
        name = ETDUMP_STEM_FLATCC,
        srcs = [],
        visibility = ["//executorch/..."],
        exported_headers = {
            ETDUMP_SCHEMA_FLATCC_BUILDER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, ETDUMP_SCHEMA_FLATCC_BUILDER),
            ETDUMP_SCHEMA_FLATCC_READER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, ETDUMP_SCHEMA_FLATCC_READER),
            ETDUMP_SCHEMA_FLATCC_VERIFIER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, ETDUMP_SCHEMA_FLATCC_VERIFIER),
            SCALAR_TYPE_BUILDER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, SCALAR_TYPE_BUILDER),
            SCALAR_TYPE_READER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, SCALAR_TYPE_READER),
            SCALAR_TYPE_VERIFIER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, SCALAR_TYPE_VERIFIER),
            FLATBUFFERS_COMMON_BUILDER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, FLATBUFFERS_COMMON_BUILDER),
            FLATBUFFERS_COMMON_READER: ":{}[{}]".format(ETDUMP_GEN_RULE_NAME_FLATCC, FLATBUFFERS_COMMON_READER),
        },
        exported_deps = [
            "fbsource//arvr/third-party/flatcc:flatcc",
        ],
    )
