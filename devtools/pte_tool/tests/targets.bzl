load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_unittest(
        name = "test_diff_pte",
        srcs = [
            "test_diff_pte.py",
        ],
        deps = [
            "//executorch/devtools/pte_tool:diff_pte_lib",
            "//executorch/exir:schema",
            "//executorch/exir/_serialize:lib",
        ],
    )
