load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # CUDA-specific AOTI functionality
    runtime.cxx_library(
        name = "aoti_cuda",
        srcs = [
            "runtime/cuda_backend.cpp",
            "runtime/shims/memory.cpp",
            "runtime/shims/tensor_attribute.cpp",
            "runtime/utils.cpp",
        ],
        headers = [
            "runtime/shims/memory.h",
            "runtime/shims/tensor_attribute.h",
            "runtime/utils.h",
        ],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "//executorch/backends/aoti:aoti_common",
            "//caffe2/torch/csrc/inductor:aoti_torch_cuda",
        ],
    )
