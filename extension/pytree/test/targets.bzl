load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    # @noautodeps


    cpp_unittest(
        name = "pytree_test",
        srcs = ["test_pytree.cpp"],
        deps = ["//executorch/extension/pytree:pytree"],
    )

    python_unittest(
        name = "pybindings_test",
        srcs = [
            "test.py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/extension/pytree:pybindings",
            "//executorch/extension/pytree:pylib",
        ],
    )
