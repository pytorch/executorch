load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")
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
        compiler_flags = ["-Wno-missing-prototypes"] + (
            # For shared library build, we don't want to expose symbols of
            # kernel implementation (ex torch::executor::native::tanh_out)
            # to library users. They should use kernels through registry only.
            # With visibility=hidden, linker won't expose kernel impl symbols
            # so it can prune unregistered kernels.
            # Currently fbcode linkes all dependent libraries through shared
            # library, and it blocks users like unit tests to use kernel
            # implementation directly. So we enable this for xplat only.
            ["-fvisibility=hidden"] if is_xplat() else []
        ),
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

def define_op_target(name, deps, android_deps, is_aten_op, is_et_op = True, _allow_third_party_deps = False, _aten_mode_deps = []):
    """Possibly defines cxx_library targets for the named operator group.

    Args:
        name: The base name of the target; e.g., "op_add"
        deps: List of deps for the targets.
        android_deps: List of fbandroid_platform_deps for the target.
        is_aten_op: True if the operator overload group is ATen-compatible.
        is_et_op: True if the operator overload group is ET-compatible.
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

    if is_et_op:
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

# Operators that are listed in `functions.yaml`, and are thus compatible with
# the core ATen operators. Every entry here will be backed by a cxx_library
# target with the given name and deps.
#
# Note that a single target (or single .cpp file) can't mix ATen and non-ATen
# ops, and must be split. They can, however, share common code via a library dep
# if necessary.
ATEN_OPS = (
    op_target(
        name = "op_abs",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_acos",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_acosh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_add",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_addmm",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":scalar_utils",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_alias_copy",
    ),
    op_target(
        name = "op_amax",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_amin",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_any",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_arange",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_argmax",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_argmin",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_as_strided_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_asin",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_asinh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_atan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_atan2",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_atanh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_avg_pool2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_bitwise_and",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/pattern:bitwise_op",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_bitwise_not",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_bitwise_or",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/pattern:bitwise_op",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_bitwise_xor",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/pattern:bitwise_op",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_bmm",
        deps = [
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_cat",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_cdist_forward",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:distance_util",
        ],
    ),
    op_target(
        name = "op_ceil",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_clamp",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
        ],
    ),
    op_target(
        name = "op_clone",
    ),
    op_target(
        name = "op_constant_pad_nd",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_convolution",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_convolution_backward",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_cos",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_cosh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_cumsum",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_detach_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_diagonal_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_div",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_embedding",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_eq",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_empty",
    ),
    op_target(
        name = "op_erf",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_exp",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_expand_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:repeat_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_expm1",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_fill",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_flip",
        deps = [
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_floor",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_floor_divide",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:math_util",
        ],
    ),
    op_target(
        name = "op_fmod",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_full",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_full_like",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_gather",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_ge",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_gelu",
        deps = [
            ":math_constants",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_glu",
        deps = [
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_gt",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_hardtanh",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_index",
        deps = [
            "//executorch/kernels/portable/cpu/util:advanced_index_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_index_put",
        deps = [
            "//executorch/kernels/portable/cpu/util:advanced_index_util",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
        ],
    ),
    op_target(
        name = "op_index_select",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_isinf",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_isnan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_le",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_leaky_relu",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_lift_fresh_copy",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_log",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log10",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log1p",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log2",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_log_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_logical_and",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logical_not",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_logical_or",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logical_xor",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_logit",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_lt",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_masked_fill",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_max",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_maximum",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_max_pool2d_with_indices",
        deps = [
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_mean",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_min",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_minimum",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:math_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_mm",
        deps = [
            "//executorch/kernels/portable/cpu/util:matmul_ops_util",
            ":vec_ops",
        ],
    ),
    op_target(
        name = "op_mul",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            ":scalar_utils",
        ],
    ),
    op_target(
        name = "op_narrow_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:slice_util",
        ],
    ),
    op_target(
        name = "op_native_batch_norm",
        deps = [
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(
        name = "op_native_group_norm",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(
        name = "op_native_layer_norm",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:normalization_ops_util",
        ],
    ),
    op_target(
        name = "op_ne",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_neg",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_nonzero",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_ones",
        deps = [
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_pdist_forward",
        deps = [
            "//executorch/kernels/portable/cpu/util:distance_util",
        ],
    ),
    op_target(
        name = "op_permute_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_pixel_shuffle",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_pixel_unshuffle",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_pow",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_prod",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_reciprocal",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_reflection_pad1d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_reflection_pad2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_reflection_pad3d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_relu",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_remainder",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:math_util",
        ],
    ),
    op_target(
        name = "op_repeat",
        deps = [
            "//executorch/kernels/portable/cpu/util:repeat_util",
        ],
    ),
    op_target(
        name = "op_replication_pad1d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_replication_pad2d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_replication_pad3d",
        deps = [
            "//executorch/kernels/portable/cpu/util:padding_util",
        ],
    ),
    op_target(
        name = "op_roll",
    ),
    op_target(
        name = "op_round",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_rsqrt",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_rsub",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_scalar_tensor",
        deps = [":scalar_utils"],
    ),
    op_target(
        name = "op_scatter",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:index_util",
        ],
    ),
    op_target(
        name = "op_scatter_add",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_select_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
            "//executorch/kernels/portable/cpu/util:select_copy_util",
        ],
    ),
    op_target(
        name = "op_select_scatter",
        deps = [
            "//executorch/kernels/portable/cpu/util:index_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
    op_target(
        name = "op_sigmoid",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_sign",
        deps = [
            "//executorch/kernels/portable/cpu/util:functional_util",
        ],
    ),
    op_target(
        name = "op_sin",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_sinh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_slice_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:slice_util",
        ],
    ),
    op_target(
        name = "op_slice_scatter",
        deps = [
            "//executorch/kernels/portable/cpu/util:slice_util",
        ],
    ),
    op_target(
        name = "op_softmax",
        deps = [
            ":vec_ops",
            "//executorch/kernels/portable/cpu/util:activation_ops_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_split_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_split_with_sizes_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_sqrt",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_squeeze_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_stack",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_sub",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/kernels/portable/cpu/util:kernel_ops_util",
        ],
    ),
    op_target(
        name = "op_sum",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_t_copy",
        deps = ["//executorch/kernels/portable/cpu/util:transpose_util"],
    ),
    op_target(
        name = "op_tan",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_tanh",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_to_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_topk",
    ),
    op_target(
        name = "op_transpose_copy",
        deps = ["//executorch/kernels/portable/cpu/util:transpose_util"],
    ),
    op_target(
        name = "op_tril",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_trunc",
        deps = [
            "//executorch/kernels/portable/cpu/pattern:pattern",
        ],
    ),
    op_target(
        name = "op_unbind_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_unsqueeze_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_var",
        deps = [
            ":scalar_utils",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/kernels/portable/cpu/util:reduce_util",
        ],
    ),
    op_target(
        name = "op_view_copy",
        deps = [
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
    op_target(
        name = "op_where",
        deps = [
            "//executorch/kernels/portable/cpu/util:broadcast_util",
            "//executorch/kernels/portable/cpu/util:functional_util",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    ),
    op_target(
        name = "op_zeros",
    ),
    op_target(
        name = "op__to_dim_order_copy",
        deps = [
            ":scalar_utils",
            "//executorch/kernels/portable/cpu/util:copy_ops_util",
        ],
    ),
)

# Operators that are not listed in `functions.yaml` (i.e., operators listed in
# `custom_ops.yaml`), which are not compatible with the core ATen operators.
# Every entry here will be backed by a cxx_library target with the given name
# and deps, as well as a similar `<name>_aten` target that uses at::Tensor and
# related types.
#
# Note that a single target (or single .cpp file) can't mix ATen and non-ATen
# ops, and must be split. They can, however, share common code via a library dep
# if necessary.
CUSTOM_OPS = (
    op_target(
        name = "op_allclose",
    ),
    op_target(
        name = "op_linear_scratch_example",
    ),
)

def portable_source_list():
    """All the source file names from //executorch/kernels/portable/cpu/"""
    return [op["name"] + ".cpp" for op in ATEN_OPS + CUSTOM_OPS]

def portable_header_list():
    """All the header file names from //executorch/kernels/portable/cpu/"""
    return ["selective_build.h", "scalar_utils.h", "math_constants.h", "vec_ops.h"]
