# Copied from fbsource/third-party/googletest

COMPILER_FLAGS = [
    "-std=c++17",
]
COMPILER_FLAGS_ATEN = [
    "-std=c++17",]

# define_gtest_targets
def define_gtest_targets():
    # Library that defines the FRIEND_TEST macro.
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "gtest_prod",
        public_system_include_directories = ["googletest/googletest/include"],
        raw_headers = ["googletest/googletest/include/gtest/gtest_prod.h"],
        visibility = ["PUBLIC"],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        # # Google Test
        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gtest" + aten_suffix,
            srcs = native.glob(
                [
                    "googletest/googletest/src/*.cc",
                ],
                exclude = [
                    "googletest/googletest/src/gtest-all.cc",
                    "googletest/googletest/src/gtest_main.cc",
                ],
            ),
            include_directories = [
                "googletest/googletest",
            ],
            public_system_include_directories = [
                "googletest/googletest/include",
            ],
            raw_headers = native.glob([
                "googletest/googletest/include/gtest/**/*.h",
                "googletest/googletest/src/*.h",
            ]),
            visibility = ["PUBLIC"],
            compiler_flags = COMPILER_FLAGS_ATEN if aten_mode else COMPILER_FLAGS,
            # TODO: gtest crashes after the test finishes when pthread is used, the root
            # cause is unclear. So it's turned off here for now. The error is as follows:
            # googletest/include/gtest/internal/gtest-port.h:1771:: pthread_key_delete(key_)failed with error 22
            exported_preprocessor_flags = ["-DGTEST_HAS_PTHREAD=0"],
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gmock" + aten_suffix,
            srcs = native.glob(
                [
                    "googletest/googlemock/src/*.cc",
                ],
                exclude = [
                    "googletest/googlemock/src/gmock-all.cc",
                    "googletest/googlemock/src/gmock_main.cc",
                ],
            ),
            include_directories = [
                "googletest/googletest",
            ],
            public_system_include_directories = [
                "googletest/googlemock/include",
            ],
            raw_headers = native.glob([
                "googletest/googlemock/include/gmock/**/*.h",
            ]),
            exported_deps = [":gtest" + aten_suffix],
            visibility = ["PUBLIC"],
            compiler_flags = COMPILER_FLAGS_ATEN if aten_mode else COMPILER_FLAGS,
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gtest_headers" + aten_suffix,
            include_directories = [
                "googletest/googletest",
            ],
            public_system_include_directories = [
                "googletest/googlemock/include",
                "googletest/googletest/include",
            ],
            raw_headers = native.glob([
                "googletest/googlemock/include/gmock/**/*.h",
                "googletest/googletest/include/gtest/**/*.h",
                "googletest/googletest/src/*.h",
            ]),
            visibility = ["PUBLIC"],
            compiler_flags = COMPILER_FLAGS_ATEN if aten_mode else COMPILER_FLAGS,
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gtest_main" + aten_suffix,
            srcs = ["googletest/googletest/src/gtest_main.cc"],
            visibility = ["PUBLIC"],
            exported_deps = [":gtest" + aten_suffix],
            compiler_flags = COMPILER_FLAGS_ATEN if aten_mode else COMPILER_FLAGS,
        )

        # # The following rules build samples of how to use gTest.
        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gtest_sample_lib" + aten_suffix,
            srcs = [
                "googletest/googletest/samples/sample1.cc",
                "googletest/googletest/samples/sample2.cc",
                "googletest/googletest/samples/sample4.cc",
            ],
            public_system_include_directories = [
                "googletest/googletest/samples",
            ],
            raw_headers = [
                "googletest/googletest/samples/prime_tables.h",
                "googletest/googletest/samples/sample1.h",
                "googletest/googletest/samples/sample2.h",
                "googletest/googletest/samples/sample3-inl.h",
                "googletest/googletest/samples/sample4.h",
            ],
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "gtest_samples" + aten_suffix,
            # All Samples except:
            #   sample9 (main)
            #   sample10 (main and takes a command line option and needs to be separate)
            srcs = [
                "googletest/googletest/samples/sample1_unittest.cc",
                "googletest/googletest/samples/sample2_unittest.cc",
                "googletest/googletest/samples/sample3_unittest.cc",
                "googletest/googletest/samples/sample4_unittest.cc",
                "googletest/googletest/samples/sample5_unittest.cc",
                "googletest/googletest/samples/sample6_unittest.cc",
                "googletest/googletest/samples/sample7_unittest.cc",
                "googletest/googletest/samples/sample8_unittest.cc",
            ],
            deps = [
                ":gtest_main" + aten_suffix,
                ":gtest_sample_lib" + aten_suffix,
            ],
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "sample9_unittest" + aten_suffix,
            srcs = ["googletest/googletest/samples/sample9_unittest.cc"],
            deps = [":gtest" + aten_suffix],
        )

        # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
        native.cxx_library(
            name = "sample10_unittest" + aten_suffix,
            srcs = ["googletest/googletest/samples/sample10_unittest.cc"],
            deps = [":gtest" + aten_suffix],
        )
