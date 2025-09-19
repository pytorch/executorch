load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Common AOTI functionality (non-CUDA)
    runtime.cxx_library(
        name = "aoti_common",
        srcs = [
            "aoti_model_container.cpp",
            "common_shims.cpp",
            "utils.cpp",
        ],
        headers = [
            "aoti_model_container.h",
            "common_shims.h",
            "utils.h",
        ],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//caffe2/torch/csrc/inductor:aoti_torch",
        ],
    )
