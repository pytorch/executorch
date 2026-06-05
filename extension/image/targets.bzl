load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

# Linker flags to pull in the Apple frameworks referenced by
# image_processor_apple_gpu.mm (CoreImage CIContext/CIImage, Foundation NS*
# classes, etc.). Applied via exported_linker_flags so they reach the final
# link of any binary/test that depends on image_processor.
_APPLE_FRAMEWORK_LINKER_FLAGS = [
    "-Wl,-framework",
    "-Wl,Accelerate",
    "-Wl,-framework",
    "-Wl,CoreGraphics",
    "-Wl,-framework",
    "-Wl,CoreImage",
    "-Wl,-framework",
    "-Wl,CoreVideo",
    "-Wl,-framework",
    "-Wl,Foundation",
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "image_processor" + aten_suffix,
            srcs = ["image_processor_common.cpp"] + select({
                "DEFAULT": ["image_processor.cpp"],
                "ovr_config//os:iphoneos": [
                    "image_processor_apple.cpp",
                    "image_processor_apple_gpu.mm",
                ],
                "ovr_config//os:macos-arm64": [
                    "image_processor_apple.cpp",
                    "image_processor_apple_gpu.mm",
                ],
            }),
            headers = [
                "image_processor_apple_gpu.h",
            ],
            exported_headers = [
                "image_processor.h",
                "image_processor_config.h",
                "image_processor_apple.h",
            ],
            visibility = ["PUBLIC"],
            deps = [
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/runtime/core:core",
            ],
            external_deps = [
                "stb",
            ],
            fbobjc_frameworks = [
                "Accelerate",
                "CoreGraphics",
                "CoreImage",
                "CoreVideo",
                "Foundation",
            ],
            # `fbobjc_frameworks` links the frameworks into this (static)
            # library but does not propagate to dependents' final link, and the
            # fbobjc_ flags don't apply on the macOS host cfg. Export the
            # framework link flags gated on the same platforms where the Apple
            # sources are compiled, so any binary/test depending on
            # image_processor links the CoreImage/Foundation/etc. symbols used
            # by image_processor_apple_gpu.mm instead of re-declaring them.
            exported_linker_flags = select({
                "DEFAULT": [],
                "ovr_config//os:iphoneos": _APPLE_FRAMEWORK_LINKER_FLAGS,
                "ovr_config//os:macos-arm64": _APPLE_FRAMEWORK_LINKER_FLAGS,
            }),
        )
