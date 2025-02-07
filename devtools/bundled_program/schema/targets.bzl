load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

BUNLDED_STEM = "bundled_program_schema"
SCALAR_TYPE_STEM = "scalar_type"

INPUT_BUNDLED = BUNLDED_STEM + ".fbs"
INPUT_SCALAR_TYPE = SCALAR_TYPE_STEM + ".fbs"

OUTPUT_BUNDLED_HEADER = BUNLDED_STEM + "_generated.h"
OUTPUT_SCALAR_TYPE_HEADER = SCALAR_TYPE_STEM + "_generated.h"

BUNDLED_GEN_RULE_NAME = "generate_bundled_program"

BUNDLED_LIBRARY_NAME = BUNLDED_STEM + "_fbs"

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
        name = INPUT_BUNDLED,
        visibility = [
            "//executorch/devtools/bundled_program/serialize/...",
        ],
    )

    runtime.export_file(
        name = INPUT_SCALAR_TYPE,
        visibility = [
            "//executorch/devtools/bundled_program/serialize/...",
        ],
    )

    _generate_schema_header(
        BUNDLED_GEN_RULE_NAME,
        [INPUT_BUNDLED, INPUT_SCALAR_TYPE],
        [OUTPUT_BUNDLED_HEADER, OUTPUT_SCALAR_TYPE_HEADER],
        OUTPUT_BUNDLED_HEADER,
    )

    # Header-only library target with the generate bundled program schema header.
    runtime.cxx_library(
        name = BUNDLED_LIBRARY_NAME,
        srcs = [],
        visibility = [
            "//executorch/devtools/bundled_program/...",
            "//executorch/extension/pybindings/...",
        ],
        exported_headers = {
            OUTPUT_BUNDLED_HEADER: ":{}[{}]".format(BUNDLED_GEN_RULE_NAME, OUTPUT_BUNDLED_HEADER),
            OUTPUT_SCALAR_TYPE_HEADER: ":{}[{}]".format(BUNDLED_GEN_RULE_NAME, OUTPUT_SCALAR_TYPE_HEADER),
        },
        exported_external_deps = ["flatbuffers-api"],
    )
