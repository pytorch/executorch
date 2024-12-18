"""Interface layer to separate Meta internal build logic and OSS build logic.

For OSS build specific changes, please add it here. For logic that needs to be shared across OSS and Meta internal, please add it to runtime_wrapper.bzl.

Also, do not load this directly from TARGETS or targets.bzl files, instead load runtime_wrapper.bzl.

"""

load(":type_defs.bzl", "is_list", "is_tuple")

_ET_TARGET_PREFIX = "executorch"

# Indicates that an external_dep entry should fall through to the underlying
# buck rule.
_EXTERNAL_DEP_FALLTHROUGH = "<fallthrough>"

_EXTERNAL_DEPS = {
    # ATen C++ library deps
    "aten-core": [],  # TODO(larryliu0820): Add support
    # ATen native_functions.yaml file deps
    "aten-src-path": "//third-party:aten_src_path",
    "cpuinfo": [],  # TODO(larryliu0820): Add support
    # Flatbuffer C++ library deps
    "flatbuffers-api": "//third-party:flatbuffers-api",
    # Flatc binary
    "flatc": "//third-party:flatc",
    # FlatCC cli binary + lib
    "flatcc": "//third-party:flatcc",
    "flatcc-cli": "//third-party:flatcc-cli",
    "flatcc-host": "//third-party:flatcc-host",
    "flatccrt": "//third-party:flatccrt",
    # Codegen driver
    "gen-executorch": "//third-party:gen_executorch",
    # Commandline flags library
    "gflags": "//third-party:gflags",
    "gmock": "//third-party:gmock",
    "gmock_aten": "//third-party:gmock_aten",
    "gtest": "//third-party:gtest",
    "gtest_aten": "//third-party:gtest_aten",
    "libtorch": "//third-party:libtorch",
    "libtorch_python": "//third-party:libtorch_python",
    "prettytable": "//third-party:prettytable",
    "pybind11": "//third-party:pybind11",
    "re2": "//extension/llm/third-party:re2",
    "sentencepiece-py": [],
    # Core C++ PyTorch functionality like Tensor and ScalarType.
    "torch-core-cpp": "//third-party:libtorch",
    "torchgen": "//third-party:torchgen",
}

def _resolve_external_dep(name):
    """Return the actual target strings for `external_deps` attribute.
    Args:
        name: The name of the dependency.
    Returns:
        A list of resolved target strings.
    """
    res = _EXTERNAL_DEPS[name]
    if is_list(res) or is_tuple(res):
        return res
    else:
        return [res]

def _start_with_et_targets(target):
    prefix = "//" + _ET_TARGET_PREFIX
    for suffix in ("/", ":"):
        if target.startswith(prefix + suffix):
            return True
    return False

def _patch_platforms(kwargs):
    """Platforms and apple_sdks are not supported yet, so pop them out from kwargs.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if "platforms" in kwargs:
        kwargs.pop("platforms")
    if "apple_sdks" in kwargs:
        kwargs.pop("apple_sdks")
    return kwargs

def _patch_deps(kwargs, dep_type):
    """Remove unsupported deps attributes from kwargs.
    dep_type: `deps`, `exported_deps`
    """
    extra_deps = kwargs.pop("fbcode_" + dep_type, [])
    kwargs.pop("xplat_" + dep_type, None)  # Also remove the other one.
    if extra_deps:
        # This should work even with select() elements.
        kwargs[dep_type] = kwargs.get(dep_type, []) + extra_deps
    return kwargs

def _patch_platform_build_mode_flags(kwargs):
    return kwargs

def _patch_force_static(kwargs):
    """For OSS cxx library, force static linkage unless specify otherwise.
    """
    if "force_static" not in kwargs:
        kwargs["force_static"] = True
    return kwargs

def _remove_platform_specific_args(kwargs):
    """Removes platform specific arguments for BUCK builds

    Args such as *_platform_preprocessor_flags and *_platform_deps are not
    supported by OSS build.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    keys = []
    for key in kwargs:
        if (key.endswith("_platform_preprocessor_flags") or key.endswith("_platform_deps") or
            key.startswith("fbobjc") or key.endswith("_platform_compiler_flags")):
            keys.append(key)
    for key in keys:
        kwargs.pop(key)
    return kwargs

def _remove_unsupported_kwargs(kwargs):
    """Removes environment unsupported kwargs
    """
    kwargs.pop("tags", None)  # tags = ["long_running"] doesn't work in oss
    kwargs.pop("types", None)  # will have to find a different way to handle .pyi files in oss
    kwargs.pop("resources", None)  # doesn't support resources in python_library/python_binary yet
    kwargs.pop("feature", None)  # internal-only, used for Product-Feature Hierarchy (PFH)
    return kwargs

def _patch_headers(kwargs):
    """Patch (add or modify or remove) headers related attributes for this build environment.
    """

    # header_namespace is to workaround the fact that all C++ source files are having the pattern:
    # `include <executorch/.../*.h>` but BUCK2 root is at executorch/ so the `executorch/` prefix is redundant.
    if "header_namespace" not in kwargs:
        kwargs["header_namespace"] = "executorch/" + native.package_name()
    return kwargs

def _patch_pp_flags(kwargs):
    return kwargs

def _patch_cxx_compiler_flags(kwargs):
    """CXX Compiler flags to enable C++17 features."""
    if "lang_compiler_flags" not in kwargs:
        kwargs["lang_compiler_flags"] = {"cxx_cpp_output": ["-std=c++17"]}
    elif "cxx_cpp_output" not in kwargs["lang_compiler_flags"]:
        kwargs["lang_compiler_flags"]["cxx_cpp_output"] = ["-std=c++17"]
    else:
        kwargs["lang_compiler_flags"]["cxx_cpp_output"].append("-std=c++17")
    return kwargs

# buildifier: disable=unused-variable
def _patch_executorch_genrule_cmd(cmd, macros_only = True):
    """Patches references to //executorch in genrule commands.

    Rewrites substrings like `//executorch/` or
    `//executorch:` and replaces them with `//` and `//:`.

    Args:
        cmd: The `cmd` string from a genrule.
        macros_only: Ignored; always treated as False.

    Returns:
        The possibly-modified command.
    """

    # Replace all references, even outside of macros.
    cmd = cmd.replace(
        "//{prefix}:".format(prefix = _ET_TARGET_PREFIX),
        ":",
    )
    cmd = cmd.replace(
        "//{prefix}/".format(prefix = _ET_TARGET_PREFIX),
        "//",
    )
    cmd = cmd.replace(
        "//xplat/{prefix}/".format(prefix = _ET_TARGET_PREFIX),
        "//",
    )
    cmd = cmd.replace(
        "fbsource//",
        "//",
    )
    return cmd

def _target_needs_patch(target):
    return _start_with_et_targets(target) or target.startswith(":")

def _patch_target_for_env(target):
    return target.replace("//executorch/", "//", 1)

def _struct_to_json(object):
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    return native.json.encode(object)

env = struct(
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    command_alias = native.command_alias,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    cxx_binary = native.cxx_binary,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    cxx_library = native.cxx_library,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    cxx_python_extension = native.cxx_python_extension,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    cxx_test = native.cxx_test,
    default_platforms = [],
    executorch_clients = ["//..."],
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    export_file = native.export_file,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    filegroup = native.filegroup,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    genrule = native.genrule,
    is_oss = True,
    is_xplat = lambda: False,
    patch_deps = _patch_deps,
    patch_cxx_compiler_flags = _patch_cxx_compiler_flags,
    patch_executorch_genrule_cmd = _patch_executorch_genrule_cmd,
    patch_force_static = _patch_force_static,
    patch_headers = _patch_headers,
    patch_platform_build_mode_flags = _patch_platform_build_mode_flags,
    patch_platforms = _patch_platforms,
    patch_pp_flags = _patch_pp_flags,
    patch_target_for_env = _patch_target_for_env,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    python_binary = native.python_binary,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    python_library = native.python_library,
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    python_test = native.python_test,
    remove_platform_specific_args = _remove_platform_specific_args,
    remove_unsupported_kwargs = _remove_unsupported_kwargs,
    resolve_external_dep = _resolve_external_dep,
    struct_to_json = _struct_to_json,
    target_needs_patch = _target_needs_patch,
    EXTERNAL_DEP_FALLTHROUGH = _EXTERNAL_DEP_FALLTHROUGH,
)
