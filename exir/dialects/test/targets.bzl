load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_unittest(
        name = "test_exir_dialect_ops",
        srcs = [
            "test_exir_dialect_ops.py",
        ],
        deps = [
            "//executorch/exir/dialects:lib",
        ],
    )
