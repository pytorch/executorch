load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    Uses fbcode-only python_unittest macro; gate on is_fbcode to preserve
    pre-migration behavior (this dir was originally TARGETS-only).
    """
    if not is_fbcode:
        return

    python_unittest(
        name = "test_llm_config",
        srcs = [
            "test_llm_config.py",
        ],
        deps = [
            "//executorch/extension/llm/export/config:llm_config",
        ],
    )
