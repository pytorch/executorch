# ------------------------------------------------------------------------------
# Add native rules to configure source files
def gflags_sources(namespace = ["google", "gflags"]):
    common_preamble = "mkdir -p `dirname $OUT` && "

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "config_h",
        exported_headers = ["gflags/src/config.h"],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.genrule(
        name = "gflags_declare_h",
        srcs = ["gflags/src/gflags_declare.h.in"],
        out = "gflags/gflags_declare.h",
        cmd = (common_preamble + "awk '{ " +
               "gsub(/@GFLAGS_NAMESPACE@/, \"" + namespace[0] + "\"); " +
               "gsub(/@(HAVE_STDINT_H|HAVE_SYS_TYPES_H|HAVE_INTTYPES_H|GFLAGS_INTTYPES_FORMAT_C99)@/, \"1\"); " +
               "gsub(/@([A-Z0-9_]+)@/, \"0\"); " +
               "print; }' $SRCS > $OUT"),
        cmd_exe = (
            "powershell " +
            "$(location :gsub_ps1) '@GFLAGS_NAMESPACE@' '" + namespace[0] + "' ${SRCS} ${TMP}/1.txt ; " +
            "$(location :gsub_ps1) '@(HAVE_STDINT_H^|HAVE_SYS_TYPES_H^|HAVE_INTTYPES_H^|GFLAGS_INTTYPES_FORMAT_C99)@' 1 ${TMP}/1.txt ${TMP}/2.txt ; " +
            "$(location :gsub_ps1) '@([A-Z0-9_]+)@' 0 ${TMP}/2.txt ${OUT}"
        ),
    )
    gflags_ns_h_files = []
    for ns in namespace[1:]:
        gflags_ns_h_file = "gflags_{}.h".format(ns)

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.genrule(
            name = gflags_ns_h_file.replace(".", "_"),
            srcs = ["gflags/src/gflags_ns.h.in"],
            out = "gflags/" + gflags_ns_h_file,
            cmd = (common_preamble + "awk '{ " +
                   "gsub(/@ns@/, \"" + ns + "\"); " +
                   "gsub(/@NS@/, \"" + ns.upper() + "\"); " +
                   "print; }' $SRCS > $OUT"),
            cmd_exe = (
                "powershell " +
                "$(location :gsub_ps1) '@ns@' '" + ns + "' ${SRCS} ${TMP}/1.txt ; " +
                "$(location :gsub_ps1) '@NS@' '" + ns.upper() + "' ${TMP}/1.txt ${OUT} "
            ),
        )
        gflags_ns_h_files.append(gflags_ns_h_file)

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.genrule(
        name = "gflags_h",
        srcs = ["gflags/src/gflags.h.in"],
        out = "gflags/gflags.h",
        cmd = (common_preamble + "awk '{ " +
               "gsub(/@GFLAGS_ATTRIBUTE_UNUSED@/, \"\"); " +
               "gsub(/@INCLUDE_GFLAGS_NS_H@/, \"" + "\n".join(["#include \\\"gflags/{}\\\"".format(hdr) for hdr in gflags_ns_h_files]) + "\"); " +
               "print; }' $SRCS > $OUT"),
        cmd_exe = (
            "powershell " +
            "$(location :gsub_ps1) '@GFLAGS_ATTRIBUTE_UNUSED@' '' ${SRCS} ${TMP}/1.txt ; " +
            "$(location :gsub_ps1) '@INCLUDE_GFLAGS_NS_H@' '" + "\n".join(["#include \\\"gflags/{}\\\"".format(hdr) for hdr in gflags_ns_h_files]) + "' ${TMP}/1.txt ${OUT} "
        ),
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.genrule(
        name = "gflags_completions_h",
        srcs = ["gflags/src/gflags_completions.h.in"],
        out = "gflags/gflags_completions.h",
        cmd = common_preamble + "awk '{ gsub(/@GFLAGS_NAMESPACE@/, \"" + namespace[0] + "\"); print; }' $SRCS > $OUT",
        cmd_exe = (
            "powershell " +
            "$(location :gsub_ps1) '@GFLAGS_NAMESPACE@' '" + namespace[0] + "' ${SRCS} ${OUT}"
        ),
    )
    headers = {
        "config.h": ":config_h",
        "mutex.h": "gflags/src/mutex.h",
        "util.h": "gflags/src/util.h",
        "windows_port.h": "gflags/src/windows_port.h",
    }
    exported_headers = {
        "gflags/gflags.h": ":gflags_h",
        "gflags/gflags_completions.h": ":gflags_completions_h",
        "gflags/gflags_declare.h": ":gflags_declare_h",
    }
    exported_headers.update({"gflags/" + hdr: ":" + hdr.replace(".", "_") for hdr in gflags_ns_h_files})
    srcs = [
        "gflags/src/gflags.cc",
        "gflags/src/gflags_completions.cc",
        "gflags/src/gflags_reporting.cc",
    ]
    return [exported_headers, headers, srcs]

# ------------------------------------------------------------------------------
# Add native rule to build gflags library
def gflags_library(name, exported_headers = {}, headers = {}, srcs = [], threads = True, deps = [], enable_static_variant = None, **kwargs):
    copts_common = [
        "-DHAVE_STDINT_H",
        "-DHAVE_SYS_TYPES_H",
        "-DHAVE_INTTYPES_H",
        "-DHAVE_SYS_STAT_H",
        "-DHAVE_UNISTD_H",
        "-DHAVE_STRTOLL",
        "-DHAVE_STRTOQ",
        "-DHAVE_RWLOCK",
        "-DGFLAGS_INTTYPES_FORMAT_C99",
        "-DGFLAGS_IS_A_DLL=0",
    ]

    copts = copts_common + [
        "-DHAVE_FNMATCH_H",
        "-DHAVE_PTHREAD",
    ]

    pthread_deps = []
    copts.append("-DNO_THREADS")

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = name,
        deps = [":_" + name],
        visibility = ["PUBLIC"],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "_" + name,
        srcs = srcs,
        headers = headers,
        header_namespace = "",
        soname = "lib{}.$(ext)".format(name),
        exported_headers = exported_headers,
        labels = [
            "depslint_never_add",  # Depslint should not add deps on these
        ],
        preprocessor_flags = copts,
        deps = deps + pthread_deps,
        # Dependents should use the public rule above.
        visibility = [],
        **kwargs
    )
