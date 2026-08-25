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
        # aten_device.h needs c10::Device and nothing else. The full libtorch
        # also registers every ATen operator, which duplicates the
        # selective-build operator library in apps that reach this target
        # through :aten_bridge, and :aten_bridge is portable mode.
        exported_external_deps = [
            "torch-core-cpp",
        ],
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
