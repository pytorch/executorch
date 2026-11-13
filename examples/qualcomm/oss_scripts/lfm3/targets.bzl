load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_library(
        name = "speech_policy",
        srcs = ["speech_policy.py"],
        base_module = "executorch.examples.qualcomm.oss_scripts.lfm3",
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "speech_policy_cpp",
        exported_headers = ["speech_policy.h"],
        visibility = ["PUBLIC"],
    )
