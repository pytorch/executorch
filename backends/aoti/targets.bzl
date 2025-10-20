load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # AOTI common shims functionality
    runtime.cxx_library(
        name = "common_shims",
        srcs = [
            "common_shims.cpp",
        ],
        headers = [
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
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )

    # AOTI model container functionality
    runtime.cxx_library(
        name = "delegate_handle",
        headers = [
            "aoti_delegate_handle.h",
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
        ],
    )

    # Common AOTI functionality (combining both common_shims and delegate_handle)
    runtime.cxx_library(
        name = "aoti_common",
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":common_shims",
            ":delegate_handle",
        ],
    )
