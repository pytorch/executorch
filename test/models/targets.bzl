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
        "ModuleAddMul",
        "ModuleBasic",
        "ModuleKVCacheCachePos",
        "ModuleKVCacheInputPos",
        "ModuleMultipleEntry",
        "ModuleNoKVCache",
        "ModuleIndex",
        "ModuleDynamicCatUnallocatedIO",
        "ModuleSimpleTrain",
        "ModuleStateful",
        "ModuleSharedState",
    ]

    # Generates Executorch .pte program files for various modules at build time.
    # To use one, depend on a target like ":exported_programs[ModuleAdd.pte]".
    runtime.genrule(
        name = "exported_programs",
        cmd = "$(exe :export_program) --modules " + ",".join(MODULES_TO_EXPORT) + " --outdir $OUT",
        outs = {
            fname + ".pte": [fname + ".pte"]
            for fname in MODULES_TO_EXPORT
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
        labels = ["reads_configerator", "justknobs"],
    )

    # Class names of nn.Modules for :exported_programs to export.
    MODULES_AND_DATA_TO_EXPORT = [
        "ModuleAddMul",
        "ModuleLinear",
        "ModuleSimpleTrain",
    ]

    runtime.genrule(
        name = "exported_program_and_data",
        cmd = "$(exe :export_program) --modules " + ",".join(MODULES_AND_DATA_TO_EXPORT) + " --external-constants --outdir $OUT",
        outs = {
            "ModuleAddMul.pte": ["ModuleAddMulProgram.pte"],
            "ModuleAddMul.ptd": ["ModuleAddMulProgram.ptd"],
            "ModuleLinear.pte": ["ModuleLinearProgram.pte"],
            "ModuleLinear.ptd": ["ModuleLinearProgram.ptd"],
            "ModuleSimpleTrainProgram.pte": ["ModuleSimpleTrainProgram.pte"],
            "ModuleSimpleTrain.ptd": ["ModuleSimpleTrainProgram.ptd"],
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
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
            "//executorch/exir/backend/test/demos/rpc:executor_backend_preprocess",
        ],
        visibility = [],  # Private
    )

    # Class names of nn.Modules available in export_delegated_program.py.
    DELEGATED_MODULES_TO_EXPORT = [
        "ModuleAddMul",
        "ModuleAddLarge",
        "ModuleSubLarge",
        "ModuleLinear",
    ]

    # Name of the backend to use when exporting delegated programs.
    BACKEND_ID = "StubBackend"

    # Generates Executorch .pte program files for the AddMul module at build time.
    # To use one, depend on a target like
    # ":exported_delegated_add_mul[ModuleAdd.pte]" or
    # ":exported_delegated_add_mul[ModuleAdd-nosegments.pte]" (which does not
    # extract the delegate data blobs into segments).
    runtime.genrule(
        name = "exported_delegated_add_mul",
        cmd = "$(exe :export_delegated_program) --modules ModuleAddMul --backend_id " + BACKEND_ID + " --outdir $OUT" +
              " && $(exe :export_delegated_program) --modules ModuleAddMul --backend_id " + BACKEND_ID + " --inline_delegate_segments --outdir $OUT" +
            # Create files with a large alignment as well as the default.
            # This alignment should be so large that it's extremely unlikely for
            # the data to accidentally be aligned to it in the default case.
              " && $(exe :export_delegated_program) --modules ModuleAddMul --backend_id " + BACKEND_ID + " --inline_delegate_segments --delegate_alignment 1024 --outdir $OUT",
        outs = {
            "ModuleAddMul.pte": ["ModuleAddMul.pte"],
            "ModuleAddMul-nosegments.pte": ["ModuleAddMul-nosegments.pte"],
            "ModuleAddMul-nosegments-da1024.pte": ["ModuleAddMul-nosegments-da1024.pte"],
        },
        default_outs = ["."],
        visibility = [
            "//executorch/runtime/executor/test/...",
            "//executorch/test/...",
        ],
    )

    runtime.genrule(
        name = "exported_xnnp_delegated_programs",
        cmd = "$(exe :export_delegated_program)" +
              " --modules ModuleAddLarge,ModuleSubLarge" +
              " --backend_id " + "XnnpackBackend" +
              " --outdir $OUT",
        outs = {
            fname + ".pte": [fname + ".pte"]
            for fname in DELEGATED_MODULES_TO_EXPORT
        },
        default_outs = ["."],
        visibility = [
            "//executorch/runtime/executor/test/...",
            "//executorch/backends/test/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        env = {"PYTORCH_DISABLE_JUSTKNOBS": "1",},
    )

    runtime.genrule(
        name = "exported_xnnpack_program_and_data",
        cmd = "$(exe :export_delegated_program)" +
            " --modules ModuleLinear" +
            " --backend_id XnnpackBackend" +
            " --external_constants" +
            " --outdir $OUT",

        outs = {
            "ModuleLinear-e.pte": ["ModuleLinear-e.pte"],
            "ModuleLinear.ptd": ["ModuleLinear.ptd"],
        },
        default_outs = ["."],
        visibility = [
            "//executorch/backends/xnnpack/test/...",
            "//executorch/test/...",
        ],
    )

    # Export with demo ExecutorBackend for program-data separation test.
    runtime.genrule(
        name = "exported_executor_backend_program_and_data",
        cmd = "$(exe :export_delegated_program)" +
            " --modules ModuleLinear" +
            " --backend_id ExecutorBackend" +
            " --external_constants" +
            " --outdir $OUT",

        outs = {
            "ModuleLinear-e.pte": ["ModuleLinear-e.pte"],
        },
        default_outs = ["."],
        visibility = [
            "//executorch/runtime/executor/test/...",
            "//executorch/extension/flat_tensor/test/...",
            "//executorch/test/...",
        ],
    )
