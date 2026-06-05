load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_test(
            name = "test" + aten_suffix,
            srcs = [
                "image_processor_test.cpp",
            ],
            deps = [
                "//executorch/extension/image:image_processor" + aten_suffix,
            ],
        )

    # Apple-specific GPU / CVPixelBuffer tests. The source is gated on
    # __APPLE__, so on non-Apple platforms this builds as an empty (passing)
    # test. CoreVideo is needed for the test's own CVPixelBuffer creation.
    runtime.cxx_test(
        name = "apple_test",
        srcs = [
            "image_processor_apple_test.cpp",
        ],
        deps = [
            "//executorch/extension/image:image_processor",
        ],
        fbobjc_frameworks = [
            "CoreVideo.framework",
        ],
    )
