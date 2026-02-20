load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.

    See README.md for instructions on selective build.
    """
    runtime.filegroup(
        name = "templates",
        srcs = native.glob([
            "templates/**/*.cpp",
            "templates/**/*.ini",
            "templates/**/*.h",
        ]),
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "macros",
        exported_headers = [
            "macros.h",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_library(
        name = "api",
        srcs = [
            "api/__init__.py",
            "api/custom_ops.py",
            "api/et_cpp.py",
            "api/types/__init__.py",
            "api/types/signatures.py",
            "api/types/types.py",
            "api/unboxing.py",
        ],
        base_module = "executorch.codegen",
        external_deps = [
            "torchgen",
        ],
    )

    runtime.python_library(
        name = "gen_lib",
        srcs = [
            "gen.py",
            "model.py",
            "parse.py",
        ],
        base_module = "executorch.codegen",
        deps = [
            ":api",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_binary(
        name = "gen",
        main_module = "executorch.codegen.gen",
        package_style = "inplace",
        deps = [
            ":gen_lib",
        ],
        _is_external_target = True,
        visibility = [
            "PUBLIC",
        ],
    )
