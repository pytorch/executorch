load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """
    Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.python_library(
        name = "linear_model",
        srcs = ["linear_model.py"],
        deps = [
            "//caffe2:torch",
        ],
        visibility = [],  # Private
    )

    runtime.python_library(
        name = "generate_linear_out_bundled_program_lib",
        srcs = ["generate_linear_out_bundled_program.py"],
        deps = [
            ":linear_model",
            "//caffe2:torch",
            "//executorch/devtools/bundled_program:config",
            "//executorch/devtools:lib",
            "//executorch/devtools/bundled_program/serialize:lib",
            "//executorch/exir:lib",
            "//executorch/exir/_serialize:lib",
        ],
    )

    runtime.python_binary(
        name = "generate_linear_out_bundled_program",
        main_module = "executorch.test.models.generate_linear_out_bundled_program",
        deps = [
            ":generate_linear_out_bundled_program_lib",
        ],
    )

    runtime.python_library(
        name = "export_program_lib",
        srcs = ["export_program.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/test/end2end:exported_module",
        ],
        visibility = [],  # Private
    )

    runtime.python_binary(
        name = "export_program",
        main_module = "executorch.test.models.export_program",
        deps = [
            ":export_program_lib",
        ],
        visibility = [],  # Private
    )

    # Class names of nn.Modules for :exported_programs to export.
    MODULES_TO_EXPORT = [
        "ModuleAdd",
        "ModuleAddHalf",
        "ModuleBasic",
        "ModuleLinear",
        "ModuleMultipleEntry",
        "ModuleIndex",
        "ModuleDynamicCatUnallocatedIO",
        "ModuleSimpleTrain",
    ]

    # Generates Executorch .pte program files for various modules at build time.
    # To use one, depend on a target like ":exported_programs[ModuleAdd.pte]".
    runtime.genrule(
        name = "exported_programs",
        cmd = "$(exe :export_program) --modules " + ",".join(MODULES_TO_EXPORT) + " --outdir $OUT",
        outs = {
            fname + seg_suffix + ".pte": [fname + seg_suffix + ".pte"]
            for fname in MODULES_TO_EXPORT
            for seg_suffix in ["", "-no-constant-segment"]
        },
        default_outs = ["."],
        visibility = [
            "//executorch/...",
            # This genrule can't run in xplat since it uses EXIR, so make its
            # output visible to xplat tests. This is an exceptional case, and
            # typically shouldn't be done.
            "fbsource//xplat/executorch/...",
        ],
        # Allow the xplat entry in the visibility list. This is an exceptional
        # case, and typically shouldn't be done.
        _is_external_target = True,
    )

    runtime.python_library(
        name = "export_delegated_program_lib",
        srcs = ["export_delegated_program.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/exir/backend:backend_api",
            "//executorch/exir/backend/test:backend_with_compiler_demo",
            "//executorch/exir:lib",
        ],
        visibility = [],  # Private
    )

    runtime.python_binary(
        name = "export_delegated_program",
        main_module = "executorch.test.models.export_delegated_program",
        # Use the https://www.internalfb.com/intern/wiki/XAR/ format so that
        # python files in the archive have predictable names/paths even in opt
        # mode. Without this `par_style` override, torch dynamo fails to skip
        # the tracing of files under the `caffe2/torch/_dynamo` directory; the
        # skips are based on the paths in the `__file__` strings at runtime, but
        # normal PAR mangles them in an incompatible way in opt mode. See
        # T151983912 for more background.
        par_style = "xar",
        deps = [
            ":export_delegated_program_lib",
        ],
        visibility = [],  # Private
    )

    # Class names of nn.Modules for :exported_delegated_programs to export.
    DELEGATED_MODULES_TO_EXPORT = [
        "ModuleAddMul",
    ]

    # Name of the backend to use when exporting delegated programs.
    BACKEND_ID = "StubBackend"

    # Generates Executorch .pte program files for various modules at build time.
    # To use one, depend on a target like
    # ":exported_delegated_programs[ModuleAdd.pte]" or
    # ":exported_delegated_programs[ModuleAdd-nosegments.pte]" (which does not
    # extract the delegate data blobs into segments).
    runtime.genrule(
        name = "exported_delegated_programs",
        cmd = "$(exe :export_delegated_program)" +
              " --modules " + ",".join(DELEGATED_MODULES_TO_EXPORT) +
              " --backend_id " + BACKEND_ID +
              " --outdir $OUT",
        outs = {
            fname + seg_suffix + da_suffix + ".pte": [fname + seg_suffix + da_suffix + ".pte"]
            for fname in DELEGATED_MODULES_TO_EXPORT
            for seg_suffix in ["", "-nosegments"]
            # "da" = delegate alignment
            for da_suffix in ["", "-da1024"]
        },
        default_outs = ["."],
        visibility = [
            "//executorch/runtime/executor/test/...",
            "//executorch/test/...",
        ],
    )
