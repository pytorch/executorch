load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "test_speech_policy_cpp",
        srcs = ["test_speech_policy.cpp"],
        deps = ["//executorch/examples/qualcomm/oss_scripts/lfm3:speech_policy_cpp"],
        external_deps = ["gtest"],
    )
