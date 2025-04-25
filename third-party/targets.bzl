load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

def define_common_targets():
    runtime.cxx_library(
        name = "unicode",
        srcs = [
            "llama.cpp-unicode/src/unicode.cpp",
            "llama.cpp-unicode/src/unicode-data.cpp",
        ],
        header_namespace = "",
        exported_headers = subdir_glob([
            ("llama.cpp-unicode/include", "*.h"),
        ]),
        visibility = ["@EXECUTORCH_CLIENTS", "//pytorch/tokenizers/..."],
    )

    if runtime.is_oss:
        runtime.cxx_library(
            name = "abseil",
            srcs = glob(
                ["abseil-cpp/absl/**/*.cc"],
                exclude = [
                    "abseil-cpp/absl/**/*test*.cc",
                    "abseil-cpp/absl/**/*mock*.cc",
                    "abseil-cpp/absl/**/*matchers*.cc",
                    "abseil-cpp/absl/**/*benchmark*.cc",
                ],
            ),
            _is_external_target = True,
            exported_linker_flags = select(
                {
                    "DEFAULT": [],
                    "ovr_config//os:macos": ["-Wl,-framework,CoreFoundation"],
                },
            ),
            public_include_directories = ["abseil-cpp"],
            visibility = ["PUBLIC"],
        )

        runtime.cxx_library(
            name = "re2",
            srcs = glob(
                [
                    "re2/re2/**/*.cc",
                    "re2/util/**/*.cc",
                ],
                exclude = [
                    "re2/re2/**/*test*.cc",
                    "re2/re2/testing/*.cc",
                    "re2/re2/fuzzing/*.cc",
                    "re2/re2/**/*benchmark*.cc",
                ],
            ),
            _is_external_target = True,
            public_include_directories = ["re2"],
            visibility = ["PUBLIC"],
            exported_deps = [
                ":abseil",
            ],
        )

        runtime.genrule(
            name = "config_h_generic",
            srcs = ["pcre2/src/config.h.generic"],
            cmd = "cp $SRCS $OUT",
            out = "pcre2/src/config.h",
        )
        runtime.genrule(
            name = "pcre2_h_generic",
            srcs = ["pcre2/src/pcre2.h.generic"],
            cmd = "cp $SRCS $OUT",
            out = "pcre2/src/pcre2.h",
        )
        runtime.genrule(
            name = "pcre2_chartables_c",
            srcs = ["pcre2/src/pcre2_chartables.c.dist"],
            cmd = "cp $SRCS $OUT",
            out = "pcre2/src/pcre2_chartables.c",
        )
        runtime.cxx_library(
            name = "pcre2",
            srcs = [
                "pcre2/src/pcre2_auto_possess.c",
                "pcre2/src/pcre2_chkdint.c",
                "pcre2/src/pcre2_compile.c",
                "pcre2/src/pcre2_compile_cgroup.c",
                "pcre2/src/pcre2_compile_class.c",
                "pcre2/src/pcre2_config.c",
                "pcre2/src/pcre2_context.c",
                "pcre2/src/pcre2_convert.c",
                "pcre2/src/pcre2_dfa_match.c",
                "pcre2/src/pcre2_error.c",
                "pcre2/src/pcre2_extuni.c",
                "pcre2/src/pcre2_find_bracket.c",
                "pcre2/src/pcre2_jit_compile.c",
                "pcre2/src/pcre2_maketables.c",
                "pcre2/src/pcre2_match.c",
                "pcre2/src/pcre2_match_data.c",
                "pcre2/src/pcre2_match_next.c",
                "pcre2/src/pcre2_newline.c",
                "pcre2/src/pcre2_ord2utf.c",
                "pcre2/src/pcre2_pattern_info.c",
                "pcre2/src/pcre2_script_run.c",
                "pcre2/src/pcre2_serialize.c",
                "pcre2/src/pcre2_string_utils.c",
                "pcre2/src/pcre2_study.c",
                "pcre2/src/pcre2_substitute.c",
                "pcre2/src/pcre2_substring.c",
                "pcre2/src/pcre2_tables.c",
                "pcre2/src/pcre2_ucd.c",
                "pcre2/src/pcre2_valid_utf.c",
                "pcre2/src/pcre2_xclass.c",
                ":pcre2_chartables_c",
            ],
            exported_headers = {"pcre2.h": ":pcre2_h_generic"},
            headers = {"config.h": ":config_h_generic"},
            # Preprocessor flags from https://github.com/PCRE2Project/pcre2/blob/2e03e323339ab692640626f02f8d8d6f95bff9c6/BUILD.bazel#L23.
            preprocessor_flags = [
                "-DHAVE_CONFIG_H",
                "-DHAVE_MEMMOVE",
                "-DHAVE_STRERROR",
                "-DPCRE2_CODE_UNIT_WIDTH=8",
                "-DPCRE2_STATIC",
                "-DSUPPORT_PCRE2_8",
                "-DSUPPORT_UNICODE",
            ] + select({
                "DEFAULT": ["-DHAVE_UNISTD_H"],
                "ovr_config//os:windows": [],
            }),
            header_namespace = "",
            _is_external_target = True,
            include_directories = ["pcre2/src"],
            visibility = ["PUBLIC"],
        )
