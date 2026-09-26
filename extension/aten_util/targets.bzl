load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Split out of :aten_bridge, which compiles in portable mode only, so that
    # ATen mode code can still convert devices.
    runtime.cxx_library(
        name = "aten_device",
        exported_headers = ["aten_device.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core/portable_type:device",
            "//executorch/runtime/platform:platform",
        ],
        # aten_device.h needs c10::Device and nothing else, and it only uses it
        # from inline functions, so plain c10 is enough. Anything wider also
        # brings in the ATen operator registry and the mobile interpreter. In
        # build environments where those live in a separate library from c10,
        # they collide at link time with the copies the application already
        # links.
        exported_external_deps = ["c10"] if runtime.is_oss else [],
        fbcode_exported_deps = [
            "fbcode//caffe2/c10:c10",
        ] if not runtime.is_oss else [],
        xplat_exported_deps = select({
            "DEFAULT": ["fbsource//xplat/caffe2/c10:c10"],
            "ovr_config//build_mode:arvr_mode[enabled]": select({
                "DEFAULT": ["fbsource//xplat/caffe2/c10:c10_ovrsource"],
                "ovr_config//os:android": ["fbsource//xplat/caffe2/c10:c10"],
            }),
        }) if not runtime.is_oss else [],
    )

    runtime.cxx_library(
        name = "aten_bridge",
        srcs = ["aten_bridge.cpp"],
        exported_headers = ["aten_bridge.h", "make_aten_functor_from_et_functor.h"],
        compiler_flags = [
            "-frtti",
            "-fno-omit-frame-pointer",
            "-fexceptions",
            "-Wno-error",
            "-Wno-unused-local-typedef",
            "-Wno-self-assign-overloaded",
            "-Wno-global-constructors",
            "-Wno-unused-function",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":aten_device",
            "//executorch/extension/kernel_util:kernel_util",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        external_deps = [
            "torch-core-cpp",
        ],
    )
