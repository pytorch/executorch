"""Common macros to build Executorch runtime targets in both fbcode and xplat.

For directories that contain code which must be built for both fbcode and xplat,
the expected pattern is to create:

- A `targets.bzl` file that uses the macros in this file
  (`runtime_wrapper.bzl`) to define a function named `define_common_targets()`.
  This function should define all of the build targets in terms of rules in the
  `runtime` struct below.

  The `targets.bzl` file must load this file from xplat (not fbcode), like
    load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
  to avoid a problematic dependency from xplat -> fbcode when building in xplat.

- A TARGETS file and a BUILD file which both contain:

    load(":targets.bzl", "define_common_targets")
    define_common_targets()

If a given directory also needs to define a fbcode-only build target as well
as the common targets, it should define that rule directly in the TARGETS file
below the call to `define_common_targets()`. Similar for xplat-only build
targets and BUCK files.

Note that fbcode-only directories do not need to use these wrappers, and can
use TARGETS files normally. Same for xplat-only directories and BUCK files.
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
load("@fbsource//tools/build_defs:selects.bzl", "selects")
load("@fbsource//tools/build_defs:type_defs.bzl", "is_dict", "is_list", "is_string", "is_tuple", "is_unicode")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load(":clients.bzl", "EXECUTORCH_CLIENTS")

# Platforms that we currently support builds for
_DEFAULT_PLATFORMS = (CXX, ANDROID, APPLE)

_DEFAULT_APPLE_SDKS = (IOS, MACOSX)

# Root directories in fbcode that we need to convert to //xplat paths.
_ET_TARGET_PREFIXES = ("executorch", "pye", "caffe2")

def get_default_executorch_platforms():
    return _DEFAULT_PLATFORMS

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

def _patch_executorch_references(targets, use_static_deps = False):
    """Patches up references to "//executorch/..." in lists of build targets.

    References to targets under `executorch` (in
    deps/exported_deps/visibility/etc.) must be specified as `//executorch/...`
    in the targets.bzl file. When building for xplat, rewrite them as
    `//xplat/executorch/...`.

    Args:
        targets: A list of build target strings to fix up. Not modified in
            place.
        use_static_deps: Whether this target should depend on static executorch
            targets when building in xplat.

    Returns:
        The possibly-different list of targets.
    """
    if not targets:
        return targets
    out_targets = []
    for target in targets:
        if target.startswith("//xplat/executorch"):
            fail("References to executorch build targets must use " +
                 "`//executorch`, not `//xplat/executorch`")
        if is_xplat():
            if _start_with_et_targets(target):
                target = target.replace("//", "//xplat/", 1)
                if use_static_deps and not target.endswith("..."):
                    target = target + "_static"
            elif use_static_deps and target.startswith(":"):
                target = target + "_static"
        out_targets.append(target)
    return out_targets

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

def _patch_build_mode_flags(kwargs):
    """Applies modifications to the `compiler_flags` kwargs based on build mode.

    Args:
        kwargs: The `kwargs` parameter from a rule.

    Returns:
        The possibly-modified `kwargs` parameter for chaining.
    """
    build_mode = native.read_config("fbcode", "build_mode_test_label", "")
    flags = []

    # Base build modes.
    if build_mode.startswith("dev"):
        flags.append("-D__ET_BUILD_MODE_DEV=1")
    elif build_mode.startswith("opt"):
        flags.append("-D__ET_BUILD_MODE_OPT=1")
    elif build_mode.startswith("dbgo"):
        flags.append("-D__ET_BUILD_MODE_DBGO=1")
    elif build_mode.startswith("dbg"):
        flags.append("-D__ET_BUILD_MODE_DBG=1")

    # Build mode extensions.
    if "-cov" in build_mode:
        flags.append("-D__ET_BUILD_MODE_COV=1")
    elif "-asan" in build_mode:
        flags.append("-D__ET_BUILD_MODE_ASAN=1")
    elif "-tsan" in build_mode:
        flags.append("-D__ET_BUILD_MODE_TSAN=1")
    elif "-ubsan" in build_mode:
        flags.append("-D__ET_BUILD_MODE_UBSAN=1")
    elif "-lto" in build_mode:
        flags.append("-D__ET_BUILD_MODE_LTO=1")

    if "compiler_flags" not in kwargs:
        kwargs["compiler_flags"] = []

    # kwargs["compiler_flags"].extend(flags) or kwargs["compiler_flags"] += would
    # fail if kwargs["compiler_flags"] is Immutable (ex: the default argument of
    # a Buck macro)
    kwargs["compiler_flags"] = kwargs["compiler_flags"] + flags

    return kwargs

def _patch_platform_build_mode_flags(kwargs):
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

def _patch_test_compiler_flags(kwargs):
    if "compiler_flags" not in kwargs:
        kwargs["compiler_flags"] = []

    # Relaxing some constraints for tests
    kwargs["compiler_flags"].extend(["-Wno-missing-prototypes", "-Wno-unused-variable", "-Wno-error"])
    return kwargs

def _patch_kwargs_common(kwargs):
    """Applies modifications to kwargs for all rule types.

    Returns the possibly-modified `kwargs` parameter for chaining.
    """

    # Be careful about dependencies on executorch targets for now, so that we
    # don't pick up unexpected clients while things are still in flux.
    if not kwargs.pop("_is_external_target", False):
        for target in kwargs.get("visibility", []):
            if not (target.startswith("//executorch") or target.startswith("@")):
                fail("Please manage all external visibility using the " +
                     "EXECUTORCH_CLIENTS list in //executorch/build/clients.bzl. " +
                     "Found external visibility target \"{}\".".format(target))
    else:
        kwargs.pop("_is_external_target", None)

    if is_xplat():
        # xplat doesn't support external_deps and exported_external_deps
        if "external_deps" in kwargs:
            kwargs.pop("external_deps")
        if "exported_external_deps" in kwargs:
            kwargs.pop("exported_external_deps")

    # Append repo-specific preprocessor_flags.
    for pp_type in ("preprocessor_flags", "exported_preprocessor_flags"):
        if is_xplat():
            extra_pp_flags = kwargs.pop("xplat_" + pp_type, [])
            kwargs.pop("fbcode_" + pp_type, None)  # Also remove the other one.
        elif is_fbcode():
            extra_pp_flags = kwargs.pop("fbcode_" + pp_type, [])
            kwargs.pop("xplat_" + pp_type, None)  # Also remove the other one.
        else:
            extra_pp_flags = []  # Silence a not-initialized warning.
            _fail_unknown_environment()
        if extra_pp_flags:
            # This should work even with select() elements.
            kwargs[pp_type] = kwargs.get(pp_type, []) + extra_pp_flags

    # Append repo-specific deps.
    for dep_type in ("deps", "exported_deps"):
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

    # Patch up references to "//executorch/..." in lists of build targets,
    # if necessary.
    use_static_deps = kwargs.pop("use_static_deps", False)
    for dep_type in ("deps", "exported_deps", "visibility"):
        if kwargs.get(dep_type):
            # deps may contain select() elements, dicts that map names to lists
            # of targets. selects.apply() will run the provided function on all
            # lists of targets in the provided object, but can also handle a
            # simple list. See also
            # https://www.internalfb.com/intern/qa/152401/what-is-a-select-in-buck
            kwargs[dep_type] = selects.apply(
                obj = kwargs.get(dep_type),
                function = native.partial(_patch_executorch_references, use_static_deps = use_static_deps),
            )

    # Make all targets private by default, like in xplat.
    if "visibility" not in kwargs:
        kwargs["visibility"] = []

    # If we see certain strings in the "visibility" list, expand them.
    if "@EXECUTORCH_CLIENTS" in kwargs["visibility"]:
        # See clients.bzl for this list.
        kwargs["visibility"].remove("@EXECUTORCH_CLIENTS")
        kwargs["visibility"].extend(EXECUTORCH_CLIENTS)

    return kwargs

def _patch_kwargs_cxx(kwargs):
    _patch_platforms(kwargs)
    _remove_platform_specific_args(kwargs)
    return _patch_kwargs_common(kwargs)

def _cxx_library_common(*args, **kwargs):
    _patch_kwargs_cxx(kwargs)
    _patch_build_mode_flags(kwargs)

    if is_xplat():
        _patch_platform_build_mode_flags(kwargs)
        fb_xplat_cxx_library(*args, **kwargs)
    elif is_fbcode():
        # fbcode doesn't support `exported_headers`; squash everything into
        # `headers`.
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
        if "exported_preprocessor_flags" in kwargs:
            kwargs["propagated_pp_flags"] = kwargs.pop(
                "exported_preprocessor_flags",
            )

        # fbcode doesn't support private dependencies.
        if "reexport_all_header_dependencies" in kwargs:
            kwargs.pop("reexport_all_header_dependencies")
        cpp_library(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _cxx_library(*args, **kwargs):
    define_static_target = kwargs.pop("define_static_target", True)

    # Determine linkage for this binary based on its children.
    kwargs["preferred_linkage"] = "any"
    _cxx_library_common(*args, **kwargs)

    # Optionally add a statically linked library target.
    if define_static_target:
        kwargs["name"] += "_static"
        kwargs["preferred_linkage"] = "static"
        kwargs["use_static_deps"] = True
        _cxx_library_common(*args, **kwargs)

def _cxx_binary_helper(*args, **kwargs):
    _patch_kwargs_cxx(kwargs)
    _patch_build_mode_flags(kwargs)
    if is_xplat():
        _patch_platform_build_mode_flags(kwargs)
        fb_xplat_cxx_binary(*args, **kwargs)
    elif is_fbcode():
        cpp_binary(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _cxx_binary(*args, **kwargs):
    define_static_target = kwargs.pop("define_static_target", True)
    _cxx_binary_helper(*args, **kwargs)
    if define_static_target:
        kwargs["name"] += "_static"
        kwargs["use_static_deps"] = True
        _cxx_binary_helper(*args, **kwargs)

def _cxx_test(*args, **kwargs):
    # Inject test utils library.
    if "deps" not in kwargs:
        kwargs["deps"] = []
    kwargs["deps"].append("//executorch/test/utils:utils")

    _patch_kwargs_cxx(kwargs)
    _patch_build_mode_flags(kwargs)
    _patch_test_compiler_flags(kwargs)

    if is_xplat():
        _patch_platform_build_mode_flags(kwargs)
        fb_xplat_cxx_test(*args, **kwargs)
    elif is_fbcode():
        cpp_unittest(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _cxx_python_extension(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        fb_cxx_python_extension(*args, **kwargs)
    elif is_fbcode():
        cpp_python_extension(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _export_file(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        # @lint-ignore BUCKLINT: fb_native is not allowed in fbcode.
        fb_native.export_file(*args, **kwargs)
    elif is_fbcode():
        export_file(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _filegroup(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        # @lint-ignore BUCKLINT: fb_native is not allowed in fbcode.
        fb_native.filegroup(*args, **kwargs)
    elif is_fbcode():
        buck_filegroup(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _genrule(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    _patch_platforms(kwargs)
    if kwargs.get("cmd"):
        kwargs["cmd"] = _patch_executorch_genrule_cmd(
            kwargs.get("cmd"),
            kwargs.pop("macros_only", True),
        )

    # Really no difference between static and non-static in genrule,
    # only to satisfy static build requirement. This is bad but works for now.
    define_static_target = kwargs.pop("define_static_target", True)
    if is_xplat():
        fb_xplat_genrule(*args, **kwargs)
    elif is_fbcode():
        buck_genrule(*args, **kwargs)
    if define_static_target:
        kwargs["name"] += "_static"
    if is_xplat():
        fb_xplat_genrule(*args, **kwargs)
    elif is_fbcode():
        buck_genrule(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _python_library(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        fb_python_library(*args, **kwargs)
    elif is_fbcode():
        python_library(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _python_binary(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        fb_python_binary(*args, **kwargs)
    elif is_fbcode():
        python_binary(*args, **kwargs)
    else:
        _fail_unknown_environment()

def _python_test(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    if is_xplat():
        fb_python_test(*args, **kwargs)
    elif is_fbcode():
        python_unittest(*args, **kwargs)
    else:
        _fail_unknown_environment()

# Names in this struct should match the standard Buck rule names if possible:
# see the "Build Rules" section in the sidebar of
# https://buck.build/concept/build_rule.html.
runtime = struct(
    cxx_binary = _cxx_binary,
    cxx_library = _cxx_library,
    cxx_python_extension = _cxx_python_extension,
    cxx_test = _cxx_test,
    export_file = _export_file,
    filegroup = _filegroup,
    genrule = _genrule,
    python_binary = _python_binary,
    python_library = _python_library,
    python_test = _python_test,
)
