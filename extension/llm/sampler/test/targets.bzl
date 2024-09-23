load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "test",
        srcs = [
            "test_sampler.cpp",
        ],
        deps = [
            "//executorch/extension/llm/sampler:sampler_aten",
        ],
        xplat_deps = [
            "//caffe2:torch_mobile_all_ops_et",
        ],
        fbcode_deps = [
            "//caffe2:torch-cpp",
        ],
    )
