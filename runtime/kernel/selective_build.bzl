load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/runtime/kernel:targets.bzl",
    "operator_registry_preprocessor_flags",
)

# Layout of the per-binary header tree, matching the angle-bracket includes
# operator_registry.cpp uses (`<executorch/runtime/kernel/...>`).
_HEADER_DIR = "executorch/runtime/kernel"
_MAX_KERNEL_NUM_HEADER = _HEADER_DIR + "/selected_max_kernel_num.h"
_OP_REGISTRY_HEADER = _HEADER_DIR + "/operator_registry.h"

def gen_max_kernel_num_genrule(
        name,
        oplist_yaml_target,
        selected_prim_ops_header_target = None,
        platforms = "CXX"):
    """Run gen_max_kernel_num on a selected_operators.yaml and emit a header
    that defines EXECUTORCH_SELECTED_MAX_KERNEL_NUM.

    When selected_prim_ops_header_target is provided (i.e. ET_PRIM_OPS_SELECTIVE_BUILD
    is active for this binary), the prim ops contribution is counted from that
    header so it matches what actually compiles in. Otherwise the count comes
    from parsing register_prim_ops.cpp directly.
    """

    # Write the header flat at the artifact root so consumers can reference
    # it as $(location :name)/selected_max_kernel_num.h (mirrors the
    # selected_operators.yaml / selected_prim_ops.h conventions used by the
    # adjacent genrules).
    cmd = (
        "$(exe //executorch/codegen/tools:gen_max_kernel_num) " +
        "--oplist-yaml=$(location {})/selected_operators.yaml ".format(oplist_yaml_target) +
        "--output-path=$OUT/selected_max_kernel_num.h "
    )
    if selected_prim_ops_header_target:
        cmd += "--selected-prim-ops-header=$(location {})/selected_prim_ops.h".format(
            selected_prim_ops_header_target,
        )
    else:
        cmd += "--prim-ops-source=$(location //executorch/kernels/prim_ops:prim_ops_sources)/register_prim_ops.cpp"

    runtime.genrule(
        name = name,
        cmd = cmd,
        outs = {"selected_max_kernel_num.h": ["selected_max_kernel_num.h"]},
        default_outs = ["."],
        platforms = platforms,
    )

def operator_registry_selective(
        name,
        selected_max_kernel_num_header_target,
        aten_suffix = "",
        platforms = "CXX",
        **kwargs):
    """Per-binary operator_registry variant whose registry capacity is sized
    to the kernels its consumer actually selected.

    Stages operator_registry.cpp + operator_registry.h + the generated
    selected_max_kernel_num.h in a single artifact tree, then compiles the
    .cpp with all three headers visible at the expected
    `<executorch/runtime/kernel/...>` paths. operator_registry.cpp's existing
    `__has_include` ladder picks up EXECUTORCH_SELECTED_MAX_KERNEL_NUM. A
    user-supplied `-c executorch.max_kernel_num=N` still wins via the same
    preprocessor flags the shared target uses.

    NOTE: the operator registry is intentionally a global; this target
    defines the same external-linkage symbols as the shared
    `//executorch/runtime/kernel:operator_registry`. Linking both into one
    binary produces ODR / duplicate-symbol errors. Consumers must arrange
    for only one to be linked transitively, which is why
    `executorch_generated_lib(auto_size_kernel_registry = True)` is opt-in.
    """
    src_target = "//executorch/runtime/kernel:operator_registry_sources"
    hdr_target = "//executorch/runtime/kernel:operator_registry_headers"
    source_name = "operator_registry.cpp"
    genrule_dep_name = name + "_operator_registry_srcs_copy"

    runtime.genrule(
        name = genrule_dep_name,
        cmd = " && ".join([
            "mkdir -p $OUT/{}".format(_HEADER_DIR),
            "cp -f $(location {})/{} $OUT/{}".format(src_target, source_name, source_name),
            "cp -f $(location {})/operator_registry.h $OUT/{}".format(hdr_target, _OP_REGISTRY_HEADER),
            "cp -f $(location {})/selected_max_kernel_num.h $OUT/{}".format(
                selected_max_kernel_num_header_target,
                _MAX_KERNEL_NUM_HEADER,
            ),
        ]),
        outs = {
            source_name: [source_name],
            "operator_registry.h": [_OP_REGISTRY_HEADER],
            "selected_max_kernel_num.h": [_MAX_KERNEL_NUM_HEADER],
        },
        default_outs = ["."],
        platforms = platforms,
    )

    runtime.cxx_library(
        name = name,
        srcs = [":" + genrule_dep_name + "[" + source_name + "]"],
        exported_headers = {
            _OP_REGISTRY_HEADER: ":" + genrule_dep_name + "[operator_registry.h]",
            _MAX_KERNEL_NUM_HEADER: ":" + genrule_dep_name + "[selected_max_kernel_num.h]",
        },
        # The dict keys above are already fully-qualified include paths (e.g.
        # `executorch/runtime/kernel/operator_registry.h`); env_interface.bzl
        # would otherwise prepend `executorch/<consumer-package>/` to them.
        header_namespace = "",
        visibility = ["PUBLIC"],
        # @lint-ignore BUCKLINT link_whole, the registry contains a global table.
        link_whole = True,
        preprocessor_flags = operator_registry_preprocessor_flags(),
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue" + aten_suffix,
        ],
        platforms = platforms,
        **kwargs
    )
