load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    fbcode_target(_kind = runtime.python_test,
        name = "test_speech_transform",
        srcs = ["test_speech_transform.py"],
        deps = [
            "//executorch/examples/models/gemma4:speech_transform",
            "fbsource//third-party/pypi/transformers:transformers",
        ],
    )
