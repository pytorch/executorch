"""This file defines and implements the interface between runtime_wrapper.bzl and environment specific logic, to prepare the proper BUCK2 rules.
It is acting as a shim layer to allow different BUCK2 rule definition such as `cxx_binary`, `cxx_library`, for different build environment: fbcode, xplat, oss.

Please maintain this file so that:
1. The only user of this file should be `runtime_wrapper.bzl`
2. Any environment specific logic should be retained inside this file and not exposed to runtime_wrapper.bzl
"""

load("@fbcode_macros//build_defs:cpp_binary.bzl", "cpp_binary")
load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:cpp_python_extension.bzl", "cpp_python_extension")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs:export_files.bzl", "export_file")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_filegroup", "buck_genrule")
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//tools/build_defs:cell_defs.bzl", "get_fbsource_cell")
load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "APPLE",
    "CXX",
    "IOS",
    "MACOSX",
)
load("@fbsource//tools/build_defs:fb_cxx_python_extension.bzl", "fb_cxx_python_extension")
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("@fbsource//tools/build_defs:fb_python_binary.bzl", "fb_python_binary")
load("@fbsource//tools/build_defs:fb_python_library.bzl", "fb_python_library")
load("@fbsource//tools/build_defs:fb_python_test.bzl", "fb_python_test")
load("@fbsource//tools/build_defs:fb_xplat_cxx_binary.bzl", "fb_xplat_cxx_binary")
load("@fbsource//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("@fbsource//tools/build_defs:fb_xplat_cxx_test.bzl", "fb_xplat_cxx_test")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode", "is_xplat")
load("@fbsource//tools/build_defs:type_defs.bzl", "is_dict", "is_list", "is_string", "is_tuple", "is_unicode")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load(":clients.bzl", "EXECUTORCH_CLIENTS")

# Unique internal keys that refer to specific repos. Values don't matter.
_FBCODE = "F"
_XPLAT = "X"

# Map of rule name to the actual rule in xplat (0th element) and fbcode (1st element)
_RULE_MAP = {
    "cxx_binary": [fb_xplat_cxx_binary, cpp_binary],
    "cxx_library": [fb_xplat_cxx_library, cpp_library],
    "cxx_python_extension": [fb_cxx_python_extension, cpp_python_extension],
    "cxx_test": [fb_xplat_cxx_test, cpp_unittest],
    # @lint-ignore BUCKLINT: fb_native is not allowed in fbcode.
    "export_file": [fb_native.export_file, export_file],
    # @lint-ignore BUCKLINT: fb_native is not allowed in fbcode.
    "filegroup": [fb_native.filegroup, buck_filegroup],
    "genrule": [fb_xplat_genrule, buck_genrule],
    "python_binary": [fb_python_binary, python_binary],
    "python_library": [fb_python_library, python_library],
    "python_test": [fb_python_test, python_unittest],
}

# Root directories in fbcode that we need to convert to //xplat paths.
_ET_TARGET_PREFIXES = ("executorch", "pye", "caffe2")

# Platforms that we currently support builds for
_DEFAULT_PLATFORMS = (CXX, ANDROID, APPLE)

_DEFAULT_APPLE_SDKS = (IOS, MACOSX)

# Indicates that an external_dep entry should fall through to the underlying
# buck rule.
_EXTERNAL_DEP_FALLTHROUGH = "<fallthrough>"

# Maps `external_deps` keys to actual targets. Values can be single strings or
# lists. If the mapped value is None, the environment (typically fbcode) will
# handle the key itself.
_EXTERNAL_DEPS_MAP = {
    # The root of a tree that contains YAML files at
    # <aten-src-patn>/aten/src/ATen/native/*.yaml
    "aten-src-path": {
        _FBCODE: "fbsource//xplat/caffe2:aten_src_path",
        _XPLAT: "fbsource//xplat/caffe2:aten_src_path",
    },
    "flatbuffers-api": {
        _FBCODE: "fbsource//third-party/flatbuffers:flatbuffers-api",
        _XPLAT: "fbsource//third-party/flatbuffers:flatbuffers-api",
    },
    # The gen_executorch commandline tool.
    "gen-executorch": {
        _FBCODE: "fbsource//xplat/caffe2/torchgen:gen_executorch",
        _XPLAT: "fbsource//xplat/caffe2/torchgen:gen_executorch",
    },
    "gflags": {
        _FBCODE: _EXTERNAL_DEP_FALLTHROUGH,
        _XPLAT: "//xplat/third-party/gflags:gflags",
    },
    "libtorch": {
        _FBCODE: "//caffe2:libtorch",
        _XPLAT: "//xplat/caffe2:torch_mobile_all_ops",
    },
}

def _start_with_et_targets(target):
    for prefix in _ET_TARGET_PREFIXES:
        prefix = "//" + prefix
        for suffix in ("/", ":"):
            if target.startswith(prefix + suffix):
                return True
    return False

def _fail_unknown_environment():
    fail("Only fbcode and xplat are supported; saw \"{}//{}\"".format(
        get_fbsource_cell(),
        native.package_name(),
    ))

def _current_repo():
    """Returns _FBCODE or _XPLAT depending on the current repo."""
    if is_fbcode():
        return _FBCODE
    elif is_xplat():
        return _XPLAT
    else:
        _fail_unknown_environment()
        return None

def _xplat_coerce_platforms(platforms):
    """Make sure `platforms` is a subset of _DEFAULT_PLATFORMS. If `platforms`
    is None, returns _DEFAULT_PLATFORMS.

    When building for xplat, platforms can be CXX, FBCODE, ANDROID, etc. But
    the only platforms ExecuTorch support currently is _DEFAULT_PLATFORMS.

    Args:
        platforms: The xplat platforms from https://fburl.com/code/fm8nq6k0

    Returns:
        The possibly-modified platforms that ExecuTorch supports.
    """
    if platforms != None:
        # if platforms is just a str/unicode, turn it into a list
        if is_string(platforms) or is_unicode(platforms):
            platforms = [platforms]

        if not is_tuple(platforms) and not is_list(platforms):
            fail("Unsupported platforms of type {}".format(type(platforms)))

        for platform in platforms:
            if platform not in _DEFAULT_PLATFORMS:
                fail("Only {} are supported; got {} instead".format(
                    _DEFAULT_PLATFORMS,
                    platforms,
                ))

        # if platforms is provided and it's a subset of _DEFAULT_PLATFORMS, then
        # it's okay to use it. Just return it.
        return platforms

    # if platforms is not provided, use the _DEFAULT_PLATFORMS
    return _DEFAULT_PLATFORMS

def _xplat_coerce_apple_sdks(platforms, apple_sdks):
    """Make sure `apple_sdks` is a subset of _DEFAULT_APPLE_SDKS. If `apple_sdks`
    is None and platforms contains APPLE, returns _DEFAULT_APPLE_SDKS.

    When building for APPLE, apple_sdks can be IOS, MACOSX, APPLETVOS, etc. But
    the only sdks ExecuTorch support currently is _DEFAULT_APPLE_SDKS.

    Args:
        platforms: The platforms for the rule we are adding apple_sdks too
        apple_sdks: The apple sdks from https://fburl.com/code/n38zqdsh

    Returns:
        The possibly-modified apple_sdks that ExecuTorch supports.
    """
    if apple_sdks != None:
        if APPLE not in platforms:
            fail("apple_sdks can only be specified if APPLE is in platforms, instead found {}".format(
                platforms,
            ))

        # if apple_sdks is just a str/unicode, turn it into a list
        if is_string(apple_sdks) or is_unicode(apple_sdks):
            apple_sdks = [apple_sdks]

        if not is_tuple(apple_sdks) and not is_list(apple_sdks):
            fail("Unsupported apple_sdks of type {}".format(type(apple_sdks)))

        for sdk in apple_sdks:
            if sdk not in _DEFAULT_APPLE_SDKS:
                fail("Only {} are supported; got {} instead".format(
                    _DEFAULT_APPLE_SDKS,
                    apple_sdks,
                ))

        # if apple_sdks is provided and it's a subset of _DEFAULT_APPLE_SDKS, then
        # it's okay to use it. Just return it.
        return apple_sdks

    # if apple_sdks is not provided, use the _DEFAULT_APPLE_SDKS
    return _DEFAULT_APPLE_SDKS if APPLE in platforms else []

def _patch_platforms(kwargs):
    """Patches platforms and apple_sdks in kwargs based on is_xplat() or is_fbcode()

    platforms and apple_sdks are only supported when building in xplat, not in fbcode. This calls
    _xplat_coerce_platforms for xplat and removes `platforms` and 'apple_sdks' for fbcode.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if is_xplat():
        kwargs["platforms"] = _xplat_coerce_platforms(kwargs.get("platforms", None))
        kwargs["apple_sdks"] = _xplat_coerce_apple_sdks(kwargs.get("platforms"), kwargs.get("apple_sdks", None))
    elif is_fbcode():
        if "platforms" in kwargs:
            kwargs.pop("platforms")
        if "apple_sdks" in kwargs:
            kwargs.pop("apple_sdks")
    else:
        _fail_unknown_environment()
    return kwargs

def _resolve_external_dep(name):
    """Converts an `external_dep` name to actual buck targets.

    Returns a sequence of targets that map to the provided `external_deps`
    name, or _EXTERNAL_DEP_FALLTHROUGH if the rule should handle the entry.
    """
    if name in _EXTERNAL_DEPS_MAP:
        target = _EXTERNAL_DEPS_MAP[name].get(_current_repo(), _EXTERNAL_DEP_FALLTHROUGH)
        if target != _EXTERNAL_DEP_FALLTHROUGH:
            # Always return a sequence of targets, even if the map only has a
            # single string.
            if is_list(target) or is_tuple(target):
                return target
            return [target]

    if is_xplat():
        # xplat doesn't support external_deps. Returning an empty list will
        # cause the caller to remove its external_deps entry.
        return []
    elif is_fbcode():
        # fbcode does support external_deps. If the name wasn't found, pass
        # it on to the underlying rule.
        return _EXTERNAL_DEP_FALLTHROUGH
    else:
        _fail_unknown_environment()
        return None

def _patch_deps(kwargs, dep_type):
    """cxx_library/cxx_binary wrapper rules supports both fbcode and xplat version of deps.
    This function extracts the deps for the proper environment and pops out the unused one.

    TODO(T158275165): Remove this once all targets use `external_deps` instead
    of repo-specific deps.

    Args:
        dep_type: can be either `deps` or `exported_deps`.
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if is_xplat():
        extra_deps = kwargs.pop("xplat_" + dep_type, [])
        kwargs.pop("fbcode_" + dep_type, None)  # Also remove the other one.
    elif is_fbcode():
        extra_deps = kwargs.pop("fbcode_" + dep_type, [])
        kwargs.pop("xplat_" + dep_type, None)  # Also remove the other one.
    else:
        extra_deps = []  # Silence a not-initialized warning.
        _fail_unknown_environment()
    if extra_deps:
        # This should work even with select() elements.
        kwargs[dep_type] = kwargs.get(dep_type, []) + extra_deps
    return kwargs

def _patch_platform_build_mode_flags(kwargs):
    """Patch xplat platform specific build mode flags.

    For xplat and ATen mode, we need to add compiler flags to support exceptions, on iOS and Android.
    """
    if is_fbcode():
        return kwargs

    # aten mode must be built with support for exceptions on ios and android.
    flags = []
    if "_aten" or "aten_" in kwargs["name"]:
        flags.append("-D__ET_ATEN=1")
        if "fbandroid_compiler_flags" not in kwargs:
            kwargs["fbandroid_compiler_flags"] = []
        if "fbobjc_compiler_flags" not in kwargs:
            kwargs["fbobjc_compiler_flags"] = []
        if "fbobjc_macosx_compiler_flags" not in kwargs:
            kwargs["fbobjc_macosx_compiler_flags"] = []
        ios_android_flags = ["-fexceptions"]
        kwargs["fbandroid_compiler_flags"].extend(ios_android_flags)
        kwargs["fbobjc_compiler_flags"].extend(ios_android_flags)
        kwargs["fbobjc_macosx_compiler_flags"].extend(ios_android_flags)
    return kwargs

def _remove_platform_specific_args(kwargs):
    """Removes platform specific arguments for FBCode builds

    Args such as *_platform_preprocessor_flags and *_platform_deps are not
    supported by FBCode builds. Remove them from kwargs if building for FBCode.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if is_fbcode():
        # Platform based Preprocessor flags
        if "cxx_platform_preprocessor_flags" in kwargs:
            kwargs.pop("cxx_platform_preprocessor_flags")
        if "fbandroid_platform_preprocessor_flags" in kwargs:
            kwargs.pop("fbandroid_platform_preprocessor_flags")

        # Platform based dependencies
        if "cxx_platform_deps" in kwargs:
            kwargs.pop("cxx_platform_deps")
        if "fbandroid_platform_deps" in kwargs:
            kwargs.pop("fbandroid_platform_deps")
    return kwargs

def _patch_headers(kwargs):
    """Add some header related handling.
    fbcode doesn't support `exported_headers` so we merge it into `headers`.
    fbcode doesn't recognize `reexport_all_header_dependencies` so pop it out.

    Modify the kwargs dict in-place, also return it.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if is_fbcode():
        if "exported_headers" in kwargs:
            exported_headers = kwargs.pop("exported_headers")

            # Note that this doesn't handle the case where one is a dict
            # and the other is a list.
            if is_dict(exported_headers):
                headers = {}
                headers.update(exported_headers)
                headers.update(kwargs.get("headers", {}))
                kwargs["headers"] = headers
            elif is_list(exported_headers):
                kwargs["headers"] = (
                    exported_headers + kwargs.get("headers", [])
                )
            else:
                fail("Unhandled exported_headers type '{}'"
                    .format(type(exported_headers)))

        # fbcode doesn't support private dependencies.
        if "reexport_all_header_dependencies" in kwargs:
            kwargs.pop("reexport_all_header_dependencies")
    return kwargs

def _patch_pp_flags(kwargs):
    """Remove unsupported preprocessing flags.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter.
    """
    if is_fbcode() and "exported_preprocessor_flags" in kwargs:
        kwargs["propagated_pp_flags"] = kwargs.pop(
            "exported_preprocessor_flags",
        )
    return kwargs

def _patch_cxx_compiler_flags(kwargs):
    return kwargs

def _patch_executorch_genrule_cmd(cmd, macros_only = True):
    """Patches references to //executorch in genrule commands.

    When building for xplat, rewrites substrings like `//executorch/` or
    `//executorch:` and replaces them with `//xplat/executorch[/:]`.

    If `macros_only` is True, only rewrites substrings in `$(exe)` or
    `$(location)` macros.

    Args:
        cmd: The `cmd` string from a genrule.
        macros_only: Only modify strings inside certain `$()` macros.

    Returns:
        The possibly-modified command.
    """
    if not cmd:
        return cmd
    if is_xplat():
        if macros_only:
            # Replace all macro references in the command. This is fragile
            # because it assumes that there is exactly one space character
            # between the macro name and the path, but it's easier to fix the
            # input than to add complexity here.
            for macro in ("location", "exe"):
                for c in (":", "/"):
                    for prefix in _ET_TARGET_PREFIXES:
                        cmd = cmd.replace(
                            "$({macro} //{prefix}{c}".format(
                                macro = macro,
                                prefix = prefix,
                                c = c,
                            ),
                            "$({macro} //xplat/{prefix}{c}".format(
                                macro = macro,
                                prefix = prefix,
                                c = c,
                            ),
                        )
        else:
            # Replace all references, even outside of macros.
            for c in (":", "/"):
                for prefix in _ET_TARGET_PREFIXES:
                    cmd = cmd.replace(
                        "//{prefix}{c}".format(prefix = prefix, c = c),
                        "//xplat/{prefix}{c}".format(prefix = prefix, c = c),
                    )
    return cmd

def _target_needs_patch(target):
    """Decide whether a target needs to be patched to satisfy the build environment.

    Args:
        target: the target string.

    Returns:
        True if the target needs to be patched, False otherwise.
    """
    return is_xplat() and (_start_with_et_targets(target) or target.startswith(":"))

def _patch_target_for_env(target):
    """For xplat, adds a prefix `//xplat/` to all targets."""
    return target.replace("//", "//xplat/", 1)

def _get_rule(rule_type, *args, **kwargs):
    """Retrieve the correct rule based on the current environment."""
    if rule_type not in _RULE_MAP:
        fail("rule {} not recognized".format(rule_type))
    rules = _RULE_MAP[rule_type]
    if is_xplat():
        return rules[0](*args, **kwargs)
    elif is_fbcode():
        return rules[1](*args, **kwargs)
    else:
        _fail_unknown_environment()
        return None

env = struct(
    cxx_binary = native.partial(_get_rule, "cxx_binary"),
    cxx_library = native.partial(_get_rule, "cxx_library"),
    cxx_python_extension = native.partial(_get_rule, "cxx_python_extension"),
    cxx_test = native.partial(_get_rule, "cxx_test"),
    default_platforms = _DEFAULT_PLATFORMS,
    executorch_clients = EXECUTORCH_CLIENTS,
    export_file = native.partial(_get_rule, "export_file"),
    filegroup = native.partial(_get_rule, "filegroup"),
    genrule = native.partial(_get_rule, "genrule"),
    patch_cxx_compiler_flags = _patch_cxx_compiler_flags,
    patch_deps = _patch_deps,
    patch_executorch_genrule_cmd = _patch_executorch_genrule_cmd,
    resolve_external_dep = _resolve_external_dep,
    patch_headers = _patch_headers,
    patch_platform_build_mode_flags = _patch_platform_build_mode_flags,
    patch_platforms = _patch_platforms,
    patch_pp_flags = _patch_pp_flags,
    patch_target_for_env = _patch_target_for_env,
    python_binary = native.partial(_get_rule, "python_binary"),
    python_library = native.partial(_get_rule, "python_library"),
    python_test = native.partial(_get_rule, "python_test"),
    remove_platform_specific_args = _remove_platform_specific_args,
    target_needs_patch = _target_needs_patch,
    EXTERNAL_DEP_FALLTHROUGH = _EXTERNAL_DEP_FALLTHROUGH,
)
