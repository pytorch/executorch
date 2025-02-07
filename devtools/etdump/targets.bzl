load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

SCALAR_TYPE_STEM = "scalar_type"
SCALAR_TYPE = SCALAR_TYPE_STEM + ".fbs"

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
            "$(exe {})".format(runtime.external_dep_location("flatcc-cli")),
            "-cwr",
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
        name = ETDUMP_SCHEMA_FLATCC,
        visibility = ["@EXECUTORCH_CLIENTS"],
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
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
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
        exported_external_deps = ["flatccrt"],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "etdump_flatcc" + aten_suffix,
            srcs = [
                "etdump_flatcc.cpp",
                "emitter.cpp",
            ],
            headers = [
                "emitter.h",
            ],
            exported_headers = [
                "etdump_flatcc.h",
            ],
            deps = [
                "//executorch/runtime/platform:platform",
            ],
            exported_deps = [
                ":etdump_schema_flatcc",
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )
