load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def prim_ops_registry_selective(name, selected_prim_ops_header_target, aten_suffix="", **kwargs):
    """
    Create a selective prim ops registry target.

    Args:
        name: Name of the target to create
        selected_prim_ops_header_target: Target that generates selected_prim_ops.h
        aten_suffix: Suffix for aten mode (e.g. "_aten")
        **kwargs: Additional arguments passed to runtime.cxx_library
    """

    target = "//executorch/kernels/prim_ops:prim_ops_sources"
    header_target = "//executorch/kernels/prim_ops:selective_build_prim_ops.h"
    source_name = "register_prim_ops.cpp"
    header_name = "selective_build_prim_ops.h"
    genrule_dep_name = name + "_register_prim_ops_srcs_copy"
    runtime.genrule(
        name = genrule_dep_name,
        cmd = "cp -f $(location {})/{} $OUT/{} && cp -f $(location {})/{} $OUT/{} && cp -f $(location {selected_prim_ops_header_target})/selected_prim_ops.h $OUT/selected_prim_ops.h".format(
            target, source_name, source_name,
            header_target, header_name, header_name,
            selected_prim_ops_header_target=selected_prim_ops_header_target
        ),
        outs = {
            source_name: [source_name],
            header_name: [header_name],
            "selected_prim_ops.h": ["selected_prim_ops.h"]
        },
        default_outs = ["."],
    )
    runtime.cxx_library(
        name = name,
        srcs = [":" + genrule_dep_name + "[register_prim_ops.cpp]"],
        exported_headers = {
            "selective_build_prim_ops.h": ":" + genrule_dep_name + "[selective_build_prim_ops.h]",
            "selected_prim_ops.h": ":" + genrule_dep_name + "[selected_prim_ops.h]"
        },
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        # @lint-ignore BUCKLINT link_whole, need this to register prim ops.
        link_whole = True,
        # prim ops are registered through a global table so the ctor needs to be allowed
        compiler_flags = select({
            "DEFAULT": ["-Wno-global-constructors"],
            "ovr_config//os:windows": [],
        }) + ["-DET_PRIM_OPS_SELECTIVE_BUILD"],
        deps = [
            "//executorch/kernels/prim_ops:et_copy_index" + aten_suffix,
            "//executorch/kernels/prim_ops:et_view" + aten_suffix,
            "//executorch/runtime/core:evalue" + aten_suffix,
            "//executorch/runtime/kernel:operator_registry" + aten_suffix,
            "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
        ],
        **kwargs
    )
