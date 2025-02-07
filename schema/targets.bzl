load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Construct the input and output file names. All input and output files rely on scalar_type file.
PROGRAM_STEM = "program"
SCALAR_TYPE_STEM = "scalar_type"

INPUT_PROGRAM = PROGRAM_STEM + ".fbs"
INPUT_SCALAR_TYPE = SCALAR_TYPE_STEM + ".fbs"

OUTPUT_PROGRAM_HEADER = PROGRAM_STEM + "_generated.h"
OUTPUT_SCALAR_TYPE_HEADER = SCALAR_TYPE_STEM + "_generated.h"

PROGRAM_GEN_RULE_NAME = "generate_program"

PROGRAM_LIRRARY_NAME = PROGRAM_STEM

def _generate_schema_header(rule_name, srcs, headers, default_header):
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

    runtime.export_file(
        name = INPUT_PROGRAM,
        visibility = [
            "//executorch/exir/_serialize/...",
        ],
    )
    runtime.export_file(
        name = INPUT_SCALAR_TYPE,
        visibility = [
            "//executorch/exir/_serialize/...",
            "//executorch/devtools/etdump/...",
        ],
    )

    _generate_schema_header(
        PROGRAM_GEN_RULE_NAME,
        [INPUT_PROGRAM, INPUT_SCALAR_TYPE],
        [OUTPUT_PROGRAM_HEADER, OUTPUT_SCALAR_TYPE_HEADER],
        OUTPUT_PROGRAM_HEADER,
    )

    # Header-only library target with the generate executorch program schema header.
    runtime.cxx_library(
        name = PROGRAM_LIRRARY_NAME,
        srcs = [],
        visibility = [
            # Lock this down as tightly as possible to ensure that flatbuffers
            # are an implementation detail. Ideally this list would only include
            # //executorch/runtime/executor/...
            "//executorch/codegen/tools/...",
            "//executorch/runtime/executor/...",
        ],
        exported_headers = {
            OUTPUT_PROGRAM_HEADER: ":{}[{}]".format(PROGRAM_GEN_RULE_NAME, OUTPUT_PROGRAM_HEADER),
            OUTPUT_SCALAR_TYPE_HEADER: ":{}[{}]".format(PROGRAM_GEN_RULE_NAME, OUTPUT_SCALAR_TYPE_HEADER),
        },
        exported_external_deps = ["flatbuffers-api"],
    )

    runtime.cxx_library(
        name = "extended_header",
        srcs = ["extended_header.cpp"],
        exported_headers = [
            "extended_header.h",
        ],
        visibility = [
            "//executorch/runtime/executor/...",
            "//executorch/schema/test/...",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )
