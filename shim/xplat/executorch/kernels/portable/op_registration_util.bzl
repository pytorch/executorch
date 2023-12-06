load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/build:selects.bzl", "selects")

def op_target(name, deps = [], android_deps = [], _allow_third_party_deps = False, _aten_mode_deps = []):
    """Registers an implementation of an operator overload group.

    An operator overload group is a set of operator overloads with a common
    operator name. That common operator name should be the base name of this
    target.

    E.g., the "add" operator overload group, named "op_add" in this target,
    might implement:
    - add.Tensor
    - add_.Tensor
    - add.out
    - add.Scalar

    If an op target would like to share a header/sources with a different op
    target (e.g., helpers/utilities), it should declare a separate cxx_library
    and add it as a dep.

    Args:
        name: The name of the operator overload group; e.g.,
            "op_add". This directory must contain a source file named
            "<name>.cpp"; e.g., "op_add.cpp".
        deps: Optional extra deps to add to the cxx_library(). Note:
            - op targets may not depend on other op targets, to keep the
              dependencies manageable. If two op targets would like to share
              code, define a separate runtime.cxx_library that they both depend
              on.
            - op targets may not depend on targets outside of `//executorch`.
              This library is intended to be portable, open-sourceable, and
              self-contained.
        android_deps: Optional extra deps to add to fb_xplat_cxx_library()
            under fbandroid_platform_deps when building for Android, which may
            be outside of //executorch. Note that these will be ignored when
            building for fbcode.
        _allow_third_party_deps: If True, the op is allowed to depend on
            third-party deps outside of //executorch. Should only be used by
            targets under //executorch/kernels/optimized, which can benefit
            from third-party optimization libraries.
        _aten_mode_deps: List of deps to add to the cxx_library() when building
            for ATen mode.
    """

    # Note that this doesn't actually define the target, but helps register
    # it in a table that's used to define the target.
    return {
        "android_deps": android_deps,
        "deps": deps,
        "name": name,
        "_allow_third_party_deps": _allow_third_party_deps,
        "_aten_mode_deps": _aten_mode_deps,
    }

def _enforce_deps(deps, name, allow_third_party_deps):
    """Fails if any of the deps are not allowed.

    Args:
        deps: A list of build target strings.
        name: The name of the target; e.g., "op_add"
        name: The name of the target with the provided deps.
        allow_third_party_deps: If True, allows external deps on third-party
            targets.
    """
    for dep in deps:
        if dep.startswith(":op_"):
            # op targets may not depend on other op targets, to keep the
            # dependencies manageable. If two op targets would like to share
            # code, define a separate runtime.cxx_library that they both depend
            # on.
            fail("op_target {} may not depend on other op_target {}".format(
                name,
                dep,
            ))
        if not (dep.startswith("//executorch") or dep.startswith(":")):
            if allow_third_party_deps and ("/third-party/" in dep):
                # Allowed exception.
                pass
            else:
                # op targets may not depend on targets outside of
                # `//executorch`. This library is intended to be portable,
                # open-sourceable, and self-contained.
                fail(
                    "op_target {} may not depend on code outside of //executorch: {}".format(
                        name,
                        dep,
                    ),
                )

def define_op_library(name, deps, android_deps, aten_target, _allow_third_party_deps = False):
    """Defines a cxx_library target for the named operator overload group.

    Args:
        name: The name of the target; e.g., "op_add"
        deps: List of deps for the target.
        android_deps: List of fbandroid_platform_deps for the target.
        aten_target: If True, define a "<name>_aten" target that uses
            `:kernel_types_aten`, compatible with host PyTorch. If False, define
            a "<name>" target that uses `:kernel_types`, compatible with the
            embedded executorch runtime.
        _allow_third_party_deps: If True, the op is allowed to depend on
            third-party deps outside of //executorch. Should only be used by
            targets under //executorch/kernels/optimized, which can benefit
            from third-party optimization libraries.
    """
    selects.apply(obj = deps, function = native.partial(_enforce_deps, name = name, allow_third_party_deps = _allow_third_party_deps))

    aten_suffix = "_aten" if aten_target else ""
    runtime.cxx_library(
        name = name + aten_suffix,
        srcs = [
            "{}.cpp".format(name),
        ],
        visibility = [
            "//executorch/kernels/portable/test/...",
            "//executorch/kernels/quantized/test/...",
            "//executorch/kernels/optimized/test/...",
            "//executorch/kernels/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        fbandroid_platform_deps = android_deps,
        # kernels often have helpers with no prototypes just disabling the warning here as the headers
        # are codegend and linked in later
        compiler_flags = ["-Wno-missing-prototypes"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
        ] + deps,
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading so library we don't want
        # it to depend on other so libraries because that way we have to
        # specify library directory path.
        force_static = True,
        # link_whole is necessary because the operators register themselves
        # via static initializers that run at program startup.
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
    )

def define_op_target(name, deps, android_deps, is_aten_op, _allow_third_party_deps = False, _aten_mode_deps = []):
    """Possibly defines cxx_library targets for the named operator group.

    Args:
        name: The base name of the target; e.g., "op_add"
        deps: List of deps for the targets.
        android_deps: List of fbandroid_platform_deps for the target.
        is_aten_op: True if the operator overload group is ATen-compatible.
        _allow_third_party_deps: If True, the op is allowed to depend on
            third-party deps outside of //executorch. Should only be used by
            targets under //executorch/kernels/optimized.
    """

    # If this is a custom op, define a target that builds it with at::Tensor
    # so that it can be imported into a host PyTorch environment for authoring.
    if not is_aten_op:
        define_op_library(
            name = name,
            deps = _aten_mode_deps if _aten_mode_deps else deps,
            android_deps = android_deps,
            aten_target = True,
            _allow_third_party_deps = _allow_third_party_deps,
        )

    # When building in ATen mode, ATen-compatible (non-custom) operators will
    # use the implementations provided by ATen, so we should not build the
    # versions defined here.
    define_op_library(
        name = name,
        deps = deps,
        android_deps = android_deps,
        aten_target = False,
        _allow_third_party_deps = _allow_third_party_deps,
    )
