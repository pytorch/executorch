load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets for the TensorRT C++ runtime backend.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "executor",
        srcs = [
            "TensorRTExecutor.cpp",
        ],
        exported_headers = [
            "TensorRTBlobHeader.h",
            "TensorRTExecutor.h",
        ],
        visibility = ["PUBLIC"],
        compiler_flags = [
            "-frtti",
            "-fexceptions",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
        fbcode_deps = [
            "fbsource//third-party/TensorRT:nvinfer-lazy",
            "//third-party-buck/platform010/build/cuda:cuda",
        ],
    )

    runtime.cxx_library(
        name = "tensorrt_backend",
        srcs = [
            "TensorRTBackend.cpp",
        ],
        exported_headers = [
            "TensorRTBackend.h",
        ],
        visibility = ["PUBLIC"],
        # Force include all symbols so the static backend registration runs.
        link_whole = True,
        compiler_flags = [
            "-Wno-global-constructors",
        ],
        deps = [
            "//executorch/runtime/backend:interface",
            ":executor",
        ],
        fbcode_deps = [
            "fbsource//third-party/TensorRT:nvinfer-lazy",
            "//third-party-buck/platform010/build/cuda:cuda",
        ],
    )
