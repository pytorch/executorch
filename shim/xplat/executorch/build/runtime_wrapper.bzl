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

load(":env_interface.bzl", "env")
load(":selects.bzl", "selects")

def is_xplat():
    return env.is_xplat()

def struct_to_json(x):
    return env.struct_to_json(struct(**x))

def get_default_executorch_platforms():
    return env.default_platforms

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

        # TODO(larryliu0820): it's confusing that we only apply "static" patch to target_needs_patch. We need to clean this up.
        if use_static_deps and env.target_needs_patch(target) and not target.endswith("..."):
            target = target + "_static"

        # Change the name of target reference to satisfy the build environment.
        # This needs to happen after any other target_needs_patch calls, because
        # it can modify the prefix of the target.
        if env.target_needs_patch(target):
            target = env.patch_target_for_env(target)

        out_targets.append(target)
    return out_targets

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

def _patch_test_compiler_flags(kwargs):
    if "compiler_flags" not in kwargs:
        kwargs["compiler_flags"] = []

    # Required globally by all c++ tests.
    kwargs["compiler_flags"].extend([
        "-std=c++17",
    ])

    # Relaxing some constraints for tests
    kwargs["compiler_flags"].extend([
        "-Wno-missing-prototypes",
        "-Wno-unused-variable",
        "-Wno-error",
    ])
    return kwargs

def _external_dep_location(name):
    """Returns the target path for the specified external_dep name.

    Only for use in genrule `$(location )`-like strings; e.g.,

        "cmd $(location " + runtime.external_dep_location("functions-yaml") + ")"

    Can only be used with external_deps names that map to exactly one target.
    """
    targets = env.resolve_external_dep(name)
    if type(targets) == type(None) or len(targets) != 1:
        fail("Could not resolve external_dep {}: saw {}".format(name, targets))
    return targets[0]

def _resolve_external_deps(kwargs):
    """Converts `[exported_]external_deps` entries into real deps if necessary."""
    for prefix in ("exported_", ""):
        external_deps = kwargs.pop(prefix + "external_deps", [])
        remaining_deps = []
        for dep in external_deps:
            targets = env.resolve_external_dep(dep)
            if targets == env.EXTERNAL_DEP_FALLTHROUGH:
                # Unhandled external_deps remain on the external_deps list of
                # the target.
                remaining_deps.append(dep)
            else:
                # Add the real targets that the external_deps entry refers to.
                if (prefix + "deps") not in kwargs:
                    kwargs[prefix + "deps"] = []

                # Do not use extend because kwargs[prefix + "deps"] may be a select.
                kwargs[prefix + "deps"] += targets
        if remaining_deps:
            kwargs[prefix + "external_deps"] = remaining_deps

def _patch_kwargs_common(kwargs):
    """Applies modifications to kwargs for all rule types.

    Returns the possibly-modified `kwargs` parameter for chaining.
    """
    env.remove_unsupported_kwargs(kwargs)

    # Be careful about dependencies on executorch targets for now, so that we
    # don't pick up unexpected clients while things are still in flux.
    if not kwargs.pop("_is_external_target", False):
        for target in kwargs.get("visibility", []):
            if not (target.startswith("//executorch") or target.startswith("@")):
                fail("Please manage all external visibility using the " +
                     "EXECUTORCH_CLIENTS list in " +
                     "//executorch/build/fb/clients.bzl. " +
                     "Found external visibility target \"{}\".".format(target))
    else:
        kwargs.pop("_is_external_target", None)

    # Convert `[exported_]external_deps` entries into real deps if necessary.
    _resolve_external_deps(kwargs)

    # TODO(T158275165): Remove this once everyone uses external_deps
    # Append repo-specific deps.
    for dep_type in ("deps", "exported_deps"):
        env.patch_deps(kwargs, dep_type)

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
        # See env.executorch_clients for this list.
        kwargs["visibility"].remove("@EXECUTORCH_CLIENTS")
        kwargs["visibility"].extend(env.executorch_clients)

    return kwargs

def _patch_kwargs_cxx(kwargs):
    env.patch_platforms(kwargs)
    env.remove_platform_specific_args(kwargs)
    return _patch_kwargs_common(kwargs)

def _cxx_library_common(*args, **kwargs):
    _patch_kwargs_cxx(kwargs)
    _patch_build_mode_flags(kwargs)

    env.patch_platform_build_mode_flags(kwargs)
    env.patch_headers(kwargs)
    env.patch_pp_flags(kwargs)
    env.patch_cxx_compiler_flags(kwargs)
    env.patch_force_static(kwargs)

    env.cxx_library(*args, **kwargs)

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
    env.patch_platform_build_mode_flags(kwargs)
    env.patch_cxx_compiler_flags(kwargs)

    env.cxx_binary(*args, **kwargs)

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

    env.patch_platform_build_mode_flags(kwargs)
    env.cxx_test(*args, **kwargs)

def _cxx_python_extension(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    kwargs["srcs"] = _patch_executorch_references(kwargs["srcs"])
    if "types" in kwargs:
        kwargs["types"] = _patch_executorch_references(kwargs["types"])
    env.cxx_python_extension(*args, **kwargs)

def _export_file(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.export_file(*args, **kwargs)

def _filegroup(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.filegroup(*args, **kwargs)

def _genrule(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.patch_platforms(kwargs)
    if kwargs.get("cmd"):
        kwargs["cmd"] = env.patch_executorch_genrule_cmd(
            kwargs.get("cmd"),
            kwargs.pop("macros_only", True),
        )

    # Really no difference between static and non-static in genrule,
    # only to satisfy static build requirement. This is bad but works for now.
    define_static_target = kwargs.pop("define_static_target", True)
    env.genrule(*args, **kwargs)
    if define_static_target:
        kwargs["name"] += "_static"
        env.genrule(*args, **kwargs)

def _python_library(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.python_library(*args, **kwargs)

def _python_binary(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.python_binary(*args, **kwargs)

def _python_test(*args, **kwargs):
    _patch_kwargs_common(kwargs)
    env.python_test(*args, **kwargs)

def get_oss_build_kwargs():
    if env.is_oss:
        return {
            "link_style": "static",
            "linker_flags": [
                # platform/system.h uses dladdr() on mac and linux
                "-ldl",
            ],
        }
    return {}

# Names in this struct should match the standard Buck rule names if possible:
# see the "Build Rules" section in the sidebar of
# https://buck.build/concept/build_rule.html.
runtime = struct(
    cxx_binary = _cxx_binary,
    cxx_library = _cxx_library,
    cxx_python_extension = _cxx_python_extension,
    cxx_test = _cxx_test,
    export_file = _export_file,
    external_dep_location = _external_dep_location,
    filegroup = _filegroup,
    genrule = _genrule,
    is_oss = env.is_oss,
    python_binary = _python_binary,
    python_library = _python_library,
    python_test = _python_test,
)
