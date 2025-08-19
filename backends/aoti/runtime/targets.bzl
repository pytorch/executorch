load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "aoti_backend",
        srcs = [
            "aoti_backend.cpp",
            "aoti_model_container.cpp",
            "shims/memory.cpp",
            "shims/tensor_attribute.cpp",
        ],
        headers = [
            "aoti_model_container.h",
            "shims/memory.h",
            "shims/tensor_attribute.h",
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
