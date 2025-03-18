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
