load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_default_executorch_platforms", "is_xplat", "runtime", "struct_to_json")
load("@fbsource//xplat/executorch/build:selects.bzl", "selects")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "portable_source_list")
load("@fbsource//xplat/executorch/kernels/optimized:op_registration_util.bzl", "optimized_source_list")
load(
    "@fbsource//xplat/executorch/kernels/optimized:lib_defs.bzl",
    "get_vec_deps",
    "get_vec_preprocessor_flags",
)

# Headers that declare the function signatures of the C++ functions that
# map to entries in functions.yaml and custom_ops.yaml.
OPERATOR_HEADERS = [
    # buildifier: keep sorted
    "Functions.h",
    "NativeFunctions.h",
]

STATIC_DISPATCH_BACKENDS = [
    "CPU",
]

# In ATen enabled mode, Functions.h will call into ATen/CPUFunctions.h and
# other ATen generated code. Hence we don't need to include the code generated
# by executorch codegen.
GENERATED_SOURCES = [
    # buildifier: keep sorted
    "RegisterCodegenUnboxedKernelsEverything.cpp",
]

MANUAL_REGISTRATION_SOURCES = [
    # buildifier: keep sorted
    "RegisterKernelsEverything.cpp",
]

MANUAL_REGISTRATION_HEADERS = [
    "RegisterKernels.h",
]

# Fake kernels only return `out` or any other tensor from arguments
CUSTOM_OPS_DUMMY_KERNEL_SOURCES = ["Register{}Stub.cpp".format(backend) for backend in STATIC_DISPATCH_BACKENDS]

CUSTOM_OPS_GENERATED_SOURCES = ["Register{}CustomOps.cpp".format(backend) for backend in STATIC_DISPATCH_BACKENDS]

CUSTOM_OPS_NATIVE_FUNCTION_HEADER = ["CustomOpsNativeFunctions.h"]

CUSTOM_OPS_SCHEMA_REGISTRATION_SOURCES = [
    "RegisterSchema.cpp",
]

ScalarType = enum(
    "Byte",
    "Char",
    "Short",
    "Int",
    "Long",
    "Half",
    "Float",
    "Double",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "Bool",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "QUInt4x2",
    "QUInt2x4",
    "Bits1x8",
    "Bits2x4",
    "Bits4x2",
    "Bits8",
    "Bits16",
    "Float8_e5m2",
    "Float8_e4m3fn",
    "Float8_e5m2fnuz",
    "Float8_e4m3fnuz",
    "UInt16",
    "UInt32",
    "Uint64",
)

# Hide the dependency to caffe2 internally.
def et_operator_library(
        name,
        ops = [],
        ops_dict = {},
        model = None,
        include_all_operators = False,
        ops_schema_yaml_target = None,
        server_generated_yaml_target = None,
        **kwargs):
    # do a dummy copy if server_generated_yaml_target is set
    if server_generated_yaml_target:
        if include_all_operators or ops_schema_yaml_target or model or ops or ops_dict:
            fail("Since server_generated_yaml_target is set, ops, ops_dict, include_all_operators and ops_schema_yaml_target shouldn't be set.")
        genrule_cmd = [
            "cp",
            "$(location {})".format(server_generated_yaml_target),
            "$OUT",
        ]
    else:
        genrule_cmd = [
            "$(exe //executorch/codegen/tools:gen_oplist)",
            "--output_path=${OUT}",
        ]
        if ops_schema_yaml_target:
            genrule_cmd.append(
                "--ops_schema_yaml_path=$(location {})".format(ops_schema_yaml_target),
            )
        if ops:
            genrule_cmd.append(
                "--root_ops=" + ",".join(ops),
            )
        if ops_dict:
            ops_dict_json = struct_to_json(ops_dict)
            genrule_cmd.append(
                "--ops_dict='{}'".format(ops_dict_json),
            )
        if model:
            genrule_cmd.append(
                "--model_file_path=$(location {})".format(model),
            )
        if include_all_operators:
            genrule_cmd.append(
                "--include_all_operators",
            )

    # TODO(larryliu0820): Remove usages of this flag.
    if "define_static_targets" in kwargs:
        kwargs.pop("define_static_targets")
    runtime.genrule(
        name = name,
        macros_only = False,
        cmd = " ".join(genrule_cmd),
        out = "selected_operators.yaml",
        labels = ["et_operator_library"],
        **kwargs
    )

def _get_headers(genrule_name, prefix = "", custom_op = None, manual_registration = False):
    headers = OPERATOR_HEADERS + (CUSTOM_OPS_NATIVE_FUNCTION_HEADER if custom_op else [])
    return {
        prefix + f: ":{}[{}]".format(genrule_name, f)
        for f in (MANUAL_REGISTRATION_HEADERS if manual_registration else [])
    }, {
        prefix + f: ":{}[{}]".format(genrule_name, f)
        for f in headers
    }

def _prepare_genrule_and_lib(
        name,
        functions_yaml_path = None,
        custom_ops_yaml_path = None,
        custom_ops_requires_runtime_registration = True,
        manual_registration = False,
        aten_mode = False,
        support_exceptions = True):
    """
    This function returns two dicts `genrules` and `libs`, derived from the arguments being passed
    to `executorch_generated_lib`. `genrules` contains all information related to what genrules to
    run. The structure of it looks like this:
    {
        <genrule_name_1>: {
            "cmd": <genrule_cmd_1>,
            "outs": <outs_list_1>,
        },
        <genrule_name_2>: {
            "cmd": <genrule_cmd_2>,
            "outs": <outs_list_2>,
        },
    }
    For `libs`, similarly it contains information related to what cxx_library we will generate.
    The structure looks like:
    {
        <lib_name_1>: {
            "genrule": <genrule_1>, # where to find the source files
            "srcs": <srcs_1>, # the source file names
        },
        <lib_name_2>: {
            "genrule": <genrule_2>, # where to find the source files
            "srcs": <srcs_2>, # the source file names
        },
    }
    """
    aten_src_path = runtime.external_dep_location("aten-src-path")
    genrule_cmd = [
        "$(exe //executorch/codegen:gen)",
        "--source-path=$(location //executorch/codegen:templates)",
        "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
        "--aten_yaml_path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
        "--install_dir=${OUT}",
        # TODO(dbort): Add a second step that verifies that the set of
        # actually-generated files matches GENERATED_FILES.
    ]

    if support_exceptions:
        genrule_cmd.append("--add-exception-boundary")


    # Sources for generated kernel registration lib
    sources = MANUAL_REGISTRATION_SOURCES if manual_registration else GENERATED_SOURCES

    # The command will always generate these files.
    genrule_outs = sources + OPERATOR_HEADERS + (CUSTOM_OPS_NATIVE_FUNCTION_HEADER if custom_ops_yaml_path else []) + MANUAL_REGISTRATION_HEADERS

    genrules = {}
    libs = {}

    # if aten_mode is true, we don't need functions_yaml_path
    genrule_name = name + "_combined"
    exported_headers, headers = _get_headers(genrule_name = genrule_name, custom_op = custom_ops_yaml_path, manual_registration = manual_registration)

    # need to register ATen ops into Executorch runtime:
    need_reg_aten_ops = aten_mode or functions_yaml_path

    # need to register custom ops into Executorch runtime:
    need_reg_custom_ops = custom_ops_yaml_path and custom_ops_requires_runtime_registration

    need_reg_ops = need_reg_aten_ops or need_reg_custom_ops

    if need_reg_aten_ops:
        path = (
            "$(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path)
        ) if not functions_yaml_path else functions_yaml_path
        genrule_cmd = genrule_cmd + [
            "--functions_yaml_path={}".format(path),
        ]
    if aten_mode:
        genrule_cmd = genrule_cmd + ["--use_aten_lib"]
    if manual_registration:
        genrule_cmd = genrule_cmd + [
            "--manual_registration",
        ]
    if custom_ops_yaml_path:
        genrule_cmd = genrule_cmd + [
            "--custom_ops_yaml_path=" + custom_ops_yaml_path,
        ]
    genrules[genrule_name] = {
        "cmd": genrule_cmd,
        "outs": genrule_outs,
    }

    if need_reg_ops:
        libs[name] = {
            "exported_headers": exported_headers,
            "genrule": genrule_name,
            "headers": headers,
            "srcs": sources,
        }

    header_lib = name + "_headers"
    libs[header_lib] = {
        "exported_headers": exported_headers,
        "headers": headers,
    }
    return genrules, libs

def _prepare_custom_ops_genrule_and_lib(
        name,
        custom_ops_yaml_path = None,
        support_exceptions = True,
        deps = [],
        kernels = []):
    """Similar to _prepare_genrule_and_lib but for custom ops."""
    genrules = {}
    libs = {}
    aten_src_path = runtime.external_dep_location("aten-src-path")
    genrule_name = name + "_gen"

    if custom_ops_yaml_path:
        # genrule for selective build from static operator list
        oplist_dir_name = name + "_oplist"
        runtime.genrule(
            name = oplist_dir_name,
            macros_only = False,
            cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:gen_all_oplist) " +
                   "--model_file_list_path $(@query_outputs 'attrfilter(labels, et_operator_library, deps(set({deps})))') " +
                   "--allow_include_all_overloads " +
                   "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
            outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
            default_outs = ["."],
        )

        # genrule for generating operator kernel bindings
        genrule_cmd = [
            "$(exe //executorch/codegen:gen)",
            "--source-path=$(location //executorch/codegen:templates)",
            "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
            "--aten_yaml_path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
            "--custom_ops_yaml_path=" + custom_ops_yaml_path,
            "--install_dir=${OUT}",
            "--op_selection_yaml_path=$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
        ]
        if support_exceptions:
            genrule_cmd.append("--add-exception-boundary")

        # Determine what sources custom_ops_<name> target should include
        custom_ops_sources = CUSTOM_OPS_SCHEMA_REGISTRATION_SOURCES + (
            CUSTOM_OPS_GENERATED_SOURCES if kernels else CUSTOM_OPS_DUMMY_KERNEL_SOURCES
        )

        # lib for registering custom ops to pytorch
        libs[name] = {
            "genrule": genrule_name,
            "headers": [],
            "srcs": custom_ops_sources,
        }
        my_cmd = ""
        for rule_substr in genrule_cmd:
            if my_cmd != "":
                my_cmd += " "
            my_cmd += rule_substr
        genrules[genrule_name] = {
            "cmd": my_cmd,
            "outs": {out: [out] for out in CUSTOM_OPS_NATIVE_FUNCTION_HEADER + custom_ops_sources},
        }
    return genrules, libs

def exir_custom_ops_aot_lib(
        name,
        yaml_target = None,
        visibility = [],
        kernels = [],
        deps = [],
        compiler_flags = [],
        define_static_target = False,
        support_exceptions = True,
        platforms = get_default_executorch_platforms()):
    """Generates a C++ library that helps to register the custom ops into PyTorch,
    so they are visible to EXIR. To use this, we need to load the generated so file:
    ```python
    torch.ops.load_library(...)
    ```

    Args:
        name: recommending a name that is obvious for user to tell this should only
            be used by EXIR (AOT) but not executorch runtime.
        yaml_target: buck target for the yaml file with proper schema and kernel entry.
            See https://github.com/pytorch/executorch/blob/main/kernels/portable/README.md#yaml-schema
            for the schema syntax.
        visibility: visibility of the generated library.
        kernels: C++ kernels for these custom ops. They need to be implemented using ATen/c10 basics.
        deps: dependencies of the generated library.
        support_exceptions: enable try/catch wrapper around operator implemntations to make sure exceptions thrown will not bring down the process. Disable if your use case disables exceptions in the build.
    """
    genrules, libs = _prepare_custom_ops_genrule_and_lib(
        name = name,
        custom_ops_yaml_path = selects.apply(yaml_target, lambda y: "$(location {})".format(y)),
        kernels = kernels,
        support_exceptions = support_exceptions,
        deps = deps,
    )
    for genrule in genrules:
        runtime.genrule(
            name = genrule,
            macros_only = False,
            cmd = genrules[genrule]["cmd"],
            outs = genrules[genrule]["outs"],
            default_outs = ["."],
        )
    for compiler_lib in libs:
        runtime.cxx_library(
            name = compiler_lib,
            srcs = [
                ":{}[{}]".format(libs[compiler_lib]["genrule"], f)
                for f in libs[compiler_lib]["srcs"]
            ],
            headers = {
                "CustomOpsNativeFunctions.h": ":{}[CustomOpsNativeFunctions.h]".format(libs[compiler_lib]["genrule"]),
            },
            # link_whole is necessary because the operators register themselves
            # via static initializers that run at program startup.
            # @lint-ignore BUCKLINT link_whole
            link_whole = True,
            visibility = visibility,
            deps = kernels + deps,
            external_deps = ["libtorch"],
            define_static_target = define_static_target,
            # Relax visibility restrictions since deps may include targets
            # outside of //executorch.
            _is_external_target = True,
            # Explicitly indicate that this C++ library will be loaded by Python
            # and consequently need to be exposed as shared libraries. It's not
            # required, but when set it'll make builds faster.
            supports_python_dlopen = True,
            platforms = platforms,
            compiler_flags = compiler_flags,
            force_static = False,
        )

def copy_files(genrule_name, target, file_list):
    """
    Copy files from `target` to current directory.
        genrule_name: name of this copy genrule.
        target: a runtime.filegroup that globs together files.
            eg. //executorch/kernels/portable/cpu:portable_source_files.
        file_list: list of filenames, used to generate the outfiles.
            eg. //executorch/kernels/portable/cpu:portable_source_list.
    """
    target_name = target.split(":")[1]
    runtime.genrule(
        name = genrule_name,
        cmd = "cp -f -r $(location {}) $OUT/".format(target),
        outs = {file: ["{}/{}".format(target_name, file)] for file in file_list},
        default_outs = ["."],
    )

def get_portable_lib_deps():
    return [
        "//executorch/kernels/portable/cpu:math_constants",
        "//executorch/kernels/portable/cpu:scalar_utils",
        "//executorch/kernels/portable/cpu:vec_ops",
        "//executorch/kernels/portable/cpu/pattern:all_deps",
        "//executorch/kernels/portable/cpu/util:all_deps",
    ]

def get_optimized_lib_deps():
    return [
        "//executorch/kernels/optimized/cpu:add_sub_impl",
        "//executorch/kernels/optimized/cpu:binary_ops",
        "//executorch/kernels/optimized/cpu:fft_utils",
        "//executorch/kernels/optimized/cpu:moments_utils",
        "//executorch/kernels/optimized:libblas",
        "//executorch/kernels/optimized:libutils",
        "//executorch/kernels/optimized:libvec",
        "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        "//executorch/runtime/kernel:kernel_includes",
    ] + get_vec_deps()

def build_portable_header_lib(name, oplist_header_name, feature = None):
    """Build the portable headers into a header-only library.
    Ensures that includes work across portable and optimized libs.
    """
    runtime.cxx_library(
        name = name,
        srcs = [],
        exported_headers = {
            "selected_op_variants.h":":{}[selected_op_variants]".format(oplist_header_name),
        },
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
        header_namespace = "",
        feature = feature,
    )

def build_portable_lib(
    name,
    et_operator_lib_deps = [],
    oplist_header_name = None,
    portable_header_lib = None,
    feature = None,
    expose_operator_symbols = False,
    visibility = ["@EXECUTORCH_CLIENTS"]):
    """
    WARNING: Before using this, please consider using executorch_generated_lib instead. This
    function is only for special cases where you need to build a portable kernel library with
    dtype selective build enabled and also wants to share it across more than one executorch_generated_lib.
    Any other use case is likely wrong and you should use executorch_generated_lib instead.

    Create a new portable kernel library based on `portable_header_lib`. `portable_header_lib`
    should contain the header `selected_op_variants.h` generated by `dtype_header_genrule`.

    Notice that this is giving a library that is different than //executorch/kernels/portable/cpu:cpu,
    because of the generated header `selected_op_variants.h`. The original portable kernel library
    doesn't have that header and thus include all the dtypes possible.

    If no `portable_header_lib` is provided, try to create one based on the deps. In this case
    we require `deps` to be present. Notice that this way we are always enabling dtype selective
    build.

    Args:
        name: name of the new portable kernel library.
        et_operator_lib_deps: list of deps to use to create the portable header library.
        oplist_header_name: the name of the header genrule (dtype_header_genrule)
        portable_header_lib: the name of the header library (build_portable_header_lib)
        feature: feature to use for the new portable kernel library.
        expose_operator_symbols: expose operator symbols to library users. This only works in xplat.
        visibility: visibility of the new portable kernel library.
    """

    if not portable_header_lib:
        if not oplist_header_name:
            if not et_operator_lib_deps:
                fail("Either et_operator_lib_deps or oplist_header_name must be provided.")
            oplist_header_name = name + "_header"
            dtype_header_genrule(
                name = oplist_header_name,
                deps = et_operator_lib_deps,
                visibility = visibility,
            )
        portable_header_lib = name + "_portable_header_lib"
        build_portable_header_lib(portable_header_lib, oplist_header_name, feature)

    # Copy portable cpp files.
    portable_source_files = []
    genrule_name = name + "_copy_portable_source"
    copy_files(genrule_name, "//executorch/kernels/portable/cpu:portable_source_files", portable_source_list())
    for op in portable_source_list():
        portable_source_files.append(":{}[{}]".format(genrule_name, op))

    # For shared library build, we don't want to expose symbols of
    # kernel implementation (ex torch::executor::native::tanh_out)
    # to library users. They should use kernels through registry only.
    # With visibility=hidden, linker won't expose kernel impl symbols
    # so it can prune unregistered kernels.
    # Currently fbcode links all dependent libraries through shared
    # library, and it blocks users like unit tests to use kernel
    # implementation directly. So we enable this for xplat only.
    compiler_flags = ["-Wno-missing-prototypes"]
    if not expose_operator_symbols and is_xplat():
        # Removing '-fvisibility=hidden' exposes operator symbols.
        # This allows operators to be called outside of the kernel registry.
        compiler_flags += ["-fvisibility=hidden"]

    # Build portable lib.
    runtime.cxx_library(
        name = name,
        srcs = portable_source_files,
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
        deps = get_portable_lib_deps() + [":" + portable_header_lib],
        compiler_flags = compiler_flags,
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading so library we don't want
        # it to depend on other so libraries because that way we have to
        # specify library directory path.
        force_static = True,
        # link_whole is necessary because the operators register themselves
        # via static initializers that run at program startup.
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        feature = feature,
    )

def build_optimized_lib(name, oplist_header_name, portable_header_lib, feature = None, expose_operator_symbols = False):
    """Build optimized lib from source. We build from source so that the generated header file,
    selected_op_variants.h, can be used to selectively build the lib for different dtypes.
    """

    # Copy optimized cpp files.
    optimized_source_files = []
    source_genrule = name + "_copy_optimized_source"
    copy_files(source_genrule, "//executorch/kernels/optimized/cpu:optimized_source_files", optimized_source_list())
    for op in optimized_source_list():
        optimized_source_files.append(":{}[{}]".format(source_genrule, op))

    # For shared library build, we don't want to expose symbols of
    # kernel implementation (ex torch::executor::native::tanh_out)
    # to library users. They should use kernels through registry only.
    # With visibility=hidden, linker won't expose kernel impl symbols
    # so it can prune unregistered kernels.
    # Currently fbcode links all dependent libraries through shared
    # library, and it blocks users like unit tests to use kernel
    # implementation directly. So we enable this for xplat only.
    compiler_flags = ["-Wno-missing-prototypes", "-Wno-pass-failed","-Wno-global-constructors","-Wno-shadow",]
    if not expose_operator_symbols and is_xplat():
        # Removing '-fvisibility=hidden' exposes operator symbols.
        # This allows operators to be called outside of the kernel registry.
        compiler_flags += ["-fvisibility=hidden"]

    # Build optimized lib.
    runtime.cxx_library(
        name = name,
        srcs = optimized_source_files,
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
        deps = get_portable_lib_deps() + get_optimized_lib_deps() + [":" + portable_header_lib],
        compiler_flags = compiler_flags,
        preprocessor_flags = get_vec_preprocessor_flags(),
        # sleef needs to be added as a direct dependency of the operator target when building for Android,
        # or a linker error may occur. Not sure why this happens; it seems that fbandroid_platform_deps of
        # dependencies are not transitive
        fbandroid_platform_deps = [
            (
                "^android-arm64.*$",
                [
                    "fbsource//third-party/sleef:sleef",
                ],
            ),
        ],
        # WARNING: using a deprecated API to avoid being built into a shared
        # library. In the case of dynamically loading so library we don't want
        # it to depend on other so libraries because that way we have to
        # specify library directory path.
        force_static = True,
        # link_whole is necessary because the operators register themselves
        # via static initializers that run at program startup.
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        feature = feature,
    )

def selected_operators_genrule(
    name,
    deps,
    platforms = get_default_executorch_platforms(),
):
    """Generates selected_operators.yaml from the list of deps. We look into the trasitive closure of all the deps,
    and look for macros `et_operator_library`.

    `gen_all_oplist` is the python binary we use to aggregate all the `et_operator_library`s into single
    `selected_operators.yaml` file.

    This file can be furthur used to generate `selected_op_variants.h` (see dtype_header_genrule) for dtype
    selective build work.
    """
    runtime.genrule(
        name = name,
        macros_only = False,
        cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:gen_all_oplist) " +
               "--model_file_list_path $(@query_outputs \'attrfilter(labels, et_operator_library, deps(set({deps})))\') " +
               "--allow_include_all_overloads " +
               "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
        outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
        default_outs = ["."],
        platforms = platforms,
    )

def dtype_header_genrule(
    name,
    visibility,
    deps = [],
    selected_operators_genrule_name = None,
    platforms = get_default_executorch_platforms(),
):
    """Generate selected_op_variants.h from selected_operators.yaml.

    Given a `selected_operators.yaml` (passed in as selected_operators_genrule_name), we should be able to determine
    what dtypes to be enabled for kernels in the kernel library. For example, `add.out` kernel needs to support
    both float16 and float32 etc.

    This information is recorded in `selected_op_variants.h` and it should be used to compile a new kernel library.

    Notice that until this stage we are kernel library agnostic, meaning the header should be applicable to any
    kernel library that includes it.
    """
    if not selected_operators_genrule_name:
        if not deps:
            fail("Either deps or selected_operators_genrule_name must be provided.")
        selected_operators_genrule_name = name + "_selected_operators"
        selected_operators_genrule(
            name = selected_operators_genrule_name,
            deps = deps,
        )

    runtime.genrule(
        name = name,
        macros_only = False,
        cmd = ("$(exe //executorch/codegen/tools:gen_selected_op_variants) " +
               "--yaml_file_path $(location :{}[selected_operators.yaml]) " +
               "--output_dir $OUT").format(selected_operators_genrule_name),
        outs = {"selected_op_variants": ["selected_op_variants.h"]},
        default_outs = ["."],
        platforms = platforms,
        visibility = visibility,
        _is_external_target = True,
    )

def executorch_generated_lib(
        name,
        functions_yaml_target = None,
        custom_ops_yaml_target = None,
        fallback_yaml_target = None,
        define_static_targets = False,
        custom_ops_aten_kernel_deps = [],
        custom_ops_requires_runtime_registration = True,
        custom_ops_requires_aot_registration = True,
        visibility = [],
        aten_mode = False,
        manual_registration = False,
        use_default_aten_ops_lib = True,
        deps = [],
        xplat_deps = [],
        fbcode_deps = [],
        platforms = get_default_executorch_platforms(),
        compiler_flags = [],
        kernel_deps = [],
        dtype_selective_build = False,
        feature = None,
        expose_operator_symbols = False,
        support_exceptions = True):
    """Emits 0-3 C++ library targets (in fbcode or xplat) containing code to
    dispatch the operators specified in the provided yaml files.

    Generates
    * `<name>` C++ library responsible to register both ATen operators and custom ops
        into Executorch runtime.
    * `custom_ops_<name>` C++ library responsible to register custom ops into PyTorch
        runtime.
    Args:
        name: The name of the C++ library target to emit. Also emits a
            header-only C++ library target named `<name>_headers` that declares
            the signatures for the C++ functions that map to the entries in
            `functions.yaml` and `custom_ops.yaml`.
            If `custom_ops_yaml_target` is specified, also emits:
            - `custom_ops_<name>`: A host-only C++ library that declares and
              registers the ops defined in that file. Clients can load this
              library into local PyTorch using `torch.ops.load_library()` to
              make them visible while authoring models.
        functions_yaml_target: A Buck target pointing to the `functions.yaml`
            file to use. Optional, but at least one of `functions_yaml_target`
            and `custom_ops_yaml_target` must be specified.
        custom_ops_yaml_target: A Buck target pointing to the `custom_ops.yaml`
            file to use. Optional, but at least one of `functions_yaml_target`
            and `custom_ops_yaml_target` must be specified.
        fallback_yaml_target: A Buck target pointing to the yaml file for fallback
            purpose. We will merge `functions.yaml` with the fallback_yaml if exist.
        define_static_targets: If True, defines extra "<name>_static" targets
            for each of the internal cxx_libraries defined by this macro, each
            with preferred_linkage="static". If false, does not define these
            targets.
        custom_ops_aten_kernel_deps: kernels for custom ops that can be registered
            into PyTorch runtime. It needs to be depending on ATen basic types such
            as `at::Tensor` and `c10::ScalarType` etc. If not provided, will auto
            generate fake kernels for custom ops.
        custom_ops_requires_runtime_registration: If false, don't generate
            `<name>` target if `functions_yaml_target` is None. If true, always
            generate `<name>` target no matter whether we have `functions_yaml_target`.
        aten_mode: a boolean for whether we should use ATen kernels and ATen tensors.
        visibility: Visibility of the C++ library targets.
        deps: Additinal deps of the main C++ library. Needs to be in either `//executorch` or `//caffe2` module.
        platforms: platforms args to runtime.cxx_library (only used when in xplat)
        manual_registration: if true, generate RegisterKernels.cpp and RegisterKernels.h.
        use_default_aten_ops_lib: If `aten_mode` is True AND this flag is True,
            use `torch_mobile_all_ops_et` for ATen operator library.
        xplat_deps: Additional xplat deps, can be used to provide custom operator library.
        fbcode_deps: Additional fbcode deps, can be used to provide custom operator library.
        compiler_flags: compiler_flags args to runtime.cxx_library
        dtype_selective_build: In additional to operator selection, dtype selective build
            further selects the dtypes for each operator. Can be used with model or dict
            selective build APIs, where dtypes can be specified.
        feature: Product-Feature Hierarchy (PFH). For internal use only, required
            for FoA in production. See: https://fburl.com/wiki/2wzjpyqy
        expose_operator_symbols: By default, fvisibility=hidden is set for executorch kernel
            libraries built with dtype selective build. This options removes the compiler
            flag and allows operators to be called outside of the kernel registry.
            NOTE: It is not recommended to set this to True, as symbols may clash (duplicate
            symbols errors) if multiple executorch_generated_libs are included by a parent library.
        support_exceptions: enable try/catch wrapper around operator implementations
            to make sure exceptions thrown will not bring down the process. Disable if your
            use case disables exceptions in the build.
    """
    if functions_yaml_target and aten_mode:
        fail("{} is providing functions_yaml_target in ATen mode, it will be ignored. `native_functions.yaml` will be the source of truth.".format(name))

    if not aten_mode and not functions_yaml_target and not custom_ops_yaml_target:
        fail("At least one of functions_yaml_target, custom_ops_yaml_target needs to be provided")

    if expose_operator_symbols:
        if not dtype_selective_build:
            fail("""
            expose_operator_symbols is only available in dtype selective build mode.
            See: https://www.internalfb.com/wiki/PyTorch/Teams/Edge/PyTorch_Edge_Core_Team/Dtype_Selective_Build/""")

    if dtype_selective_build:
        if not expose_operator_symbols and not (is_xplat() or runtime.is_oss):
            fail("""
                Dtype selective build with expose_operator_symbols=False works only in xplat -
                there are undefined symbols otherwise. Please try to use xplat, or talk to the
                executorch team. Setting expose_operator_symbols=True is not recommended as the
                exposed symbols may clash (duplicate symbols errors) if multiple
                executorch_generated_libs are included by a parent library.

                Falling back to operator selective build.""")

        if (not "//executorch/kernels/portable:operators" in kernel_deps) and (not "//executorch/kernels/optimized:optimized_operators" in kernel_deps):
            fail("""
            !!WARNING!! Dtype selective build is available for the portable and optimized kernel libraries.
            If you are using those, please add them to `kernel_deps` in `executorch_generated_lib`:
            //executorch/kernels/portable:operators
            //executorch/kernels/optimized:optimized_operators
            This will tell the build system to rebuild portable/optimized with the dtype selective build header.
            For examples, see: //executorch/examples/selective_build/targets.bzl
            Currently, kernel_deps contains {}.

            If you have a custom kernel library, please remove `dtype_selective_build=True`
            and use regular selective build.
            """.format(kernel_deps))

        # Dtype selective build requires that the portable/optimized kernel libraries are not passed into `deps`.
        if ("//executorch/kernels/portable:operators" in kernel_deps):
            index = 0
            for dep in deps:
                index = index + 1
                portable = name + "_check_portable_" + dep.split(":")[1] + str(index)
                message = "Dtype selective build requires that the portable library is not passed into `deps`. This will cause duplicate symbol errors in the build. Please remove it from `deps` and place it into `kernel_deps`"
                check_recursive_dependencies(portable, dep, "//executorch/kernels/portable:operators", message)
        if ("//executorch/kernels/optimized:optimized_operators" in kernel_deps):
            index = 0
            for dep in deps:
                index = index + 1
                optimized = name + "_check_optimized_" + dep.split(":")[1] + str(index)
                message = "Dtype selective build requires that the optimized library is not passed into `deps`. This will cause duplicate symbol errors in the build. Please remove it from `deps` and place it into `kernel_deps`"
                check_recursive_dependencies(optimized, dep, "//executorch/kernels/optimized:optimized_operators", message)


    aten_suffix = "_aten" if aten_mode else ""

    # merge functions.yaml with fallback yaml
    if functions_yaml_target:
        merge_yaml_name = name + "_merge_yaml"
        cmd = selects.apply(functions_yaml_target, lambda value: "$(exe fbsource//xplat/executorch/codegen/tools:merge_yaml) " +
                                                                 "--functions_yaml_path=$(location {}) --output_dir=$OUT ".format(value))
        if fallback_yaml_target:
            cmd = cmd + "--fallback_yaml_path=$(location {}) ".format(fallback_yaml_target)
        runtime.genrule(
            name = merge_yaml_name,
            macros_only = False,
            cmd = cmd,
            outs = {"merged.yaml": ["merged.yaml"]},
            default_outs = ["."],
            platforms = platforms,
        )
        functions_yaml_path = "$(location :{}[merged.yaml])".format(merge_yaml_name)
    else:
        functions_yaml_path = None
    if custom_ops_yaml_target:
        custom_ops_yaml_path = selects.apply(custom_ops_yaml_target, lambda value: "$(location {})".format(value))
    else:
        custom_ops_yaml_path = None

    genrules, libs = _prepare_genrule_and_lib(
        name = name,
        functions_yaml_path = functions_yaml_path,
        custom_ops_yaml_path = custom_ops_yaml_path,
        custom_ops_requires_runtime_registration = custom_ops_requires_runtime_registration,
        aten_mode = aten_mode,
        manual_registration = manual_registration,
        support_exceptions = support_exceptions,
    )

    # genrule for selective build from static operator list
    oplist_dir_name = name + "_et_oplist"
    selected_operators_genrule(name = oplist_dir_name, deps = deps, platforms = platforms)

    # genrule to generate selected_op_variants.h from selected_operators.yaml above
    oplist_header_name = name + "_et_op_dtype_gen"
    dtype_header_genrule(name = oplist_header_name, selected_operators_genrule_name = oplist_dir_name, platforms = platforms, visibility = visibility)

    # codegen genrule(s). For ATen mode we expect two genrules, one for ATen ops one for custom ops.
    for genrule_name in genrules:
        genrules[genrule_name]["cmd"].append(
            "--op_selection_yaml_path=$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
        )
        my_cmd = ""
        for rule_substr in genrules[genrule_name]["cmd"]:
            if my_cmd != "":
                my_cmd += " "
            my_cmd += rule_substr
        runtime.genrule(
            name = genrule_name,
            cmd = my_cmd,
            outs = {f: [f] for f in genrules[genrule_name]["outs"]},
            default_outs = ["."],
            platforms = platforms,
        )

    if dtype_selective_build:
        # Build portable headers lib. Used for portable and optimized kernel libraries.
        portable_header_lib = name + "_portable_header_lib"
        build_portable_header_lib(portable_header_lib, oplist_header_name, feature)

        if "//executorch/kernels/portable:operators" in kernel_deps:
            # Remove portable from kernel_deps as we're building it from source.
            kernel_deps.remove("//executorch/kernels/portable:operators")

            # Build portable lib.
            portable_lib_name = name + "_portable_lib"
            build_portable_lib(name = portable_lib_name, portable_header_lib = portable_header_lib, feature = feature, expose_operator_symbols = expose_operator_symbols)
            kernel_deps.append(":{}".format(portable_lib_name))

        if "//executorch/kernels/optimized:optimized_operators" in kernel_deps:
            # Remove optimized from kernel_deps as we're building it from source.
            kernel_deps.remove("//executorch/kernels/optimized:optimized_operators")

            # Build optimized lib.
            optimized_lib_name = name + "_optimized_lib"
            build_optimized_lib(optimized_lib_name, oplist_header_name, portable_header_lib, feature, expose_operator_symbols)
            kernel_deps.append(":{}".format(optimized_lib_name))

    # Exports headers that declare the function signatures of the C++ functions
    # that map to entries in `functions.yaml` and `custom_ops.yaml`.
    # For ATen mode, the headers will be `aten_Functions.h`, `aten_NativeFunctions.h` and `aten_UnboxingFunctions.h`
    # along with headers declaring custom ops `Functions.h`, `NativeFunctions.h` and `UnboxingFunctions.h`.
    header_lib = name + "_headers"
    if header_lib in libs:
        runtime.cxx_library(
            name = header_lib,
            srcs = [],
            exported_headers = libs[header_lib]["headers"],
            visibility = visibility,
            # Relax visibility restrictions since deps may include targets
            # outside of //executorch.
            _is_external_target = True,
            platforms = platforms,
            compiler_flags = compiler_flags,
            exported_deps = [
                "//executorch/codegen:macros",
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            feature = feature,
        )

    if name in libs:
        lib_name = name
        runtime.cxx_library(
            name = lib_name,
            srcs = [
                ":{}[{}]".format(libs[lib_name]["genrule"], f)
                for f in libs[lib_name]["srcs"]
            ],
            # Note that all of these generated headers are only used by this library
            # target, and are not meant to be used by targets outside of this
            # directory.
            headers = libs[lib_name]["headers"],
            exported_headers = libs[lib_name]["exported_headers"],
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            # link_whole is necessary because the operators register themselves via
            # static initializers that run at program startup.
            # @lint-ignore BUCKLINT link_whole
            link_whole = True,
            visibility = visibility,
            # Operator Registration is done through static tables
            compiler_flags = select({
                "DEFAULT": ["-Wno-global-constructors"],
                "ovr_config//os:windows": [],
            }) + compiler_flags,
            deps = [
                "//executorch/runtime/kernel:operator_registry" + aten_suffix,
                "//executorch/kernels/prim_ops:prim_ops_registry" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/codegen:macros",
            ] + deps + kernel_deps,
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            xplat_deps = xplat_deps,
            fbcode_deps = fbcode_deps,
            external_deps = ["libtorch"] if aten_mode and use_default_aten_ops_lib else [],
            define_static_target = define_static_targets,
            # Relax visibility restrictions since deps may include targets outside
            # of //executorch.
            _is_external_target = True,
            platforms = platforms,
            feature = feature,
        )

    if custom_ops_yaml_target and custom_ops_requires_aot_registration:
        exir_custom_ops_aot_lib(
            name = "custom_ops_" + name,
            yaml_target = custom_ops_yaml_target,
            visibility = visibility,
            kernels = custom_ops_aten_kernel_deps,
            deps = deps + [":" + header_lib],
            define_static_target = define_static_targets,
            platforms = platforms,
        )

# Util macro that takes in a binary or a shared library, find targets ending with `_et_oplist` in the transitive closure of deps,
# get the `selected_operators.yaml` from those targets, try to merge them into a single yaml. This target will fail to build, if
# there are intersections of all `selected_operators.yaml` the `target` is depending on.
#
# An example failure case: a binary `bin` is depending on 2 `executorch_generated_lib`s and they both register `aten::add.out`
# with either the same or different kernels associated to it.
#
# If build successfully, all of the `selected_operators.yaml` will be merged into 1 `selected_operators.yaml` for debugging purpose.
def executorch_ops_check(
    name,
    deps,
    **kwargs,
):
    runtime.genrule(
        name = name,
        macros_only = False,
        cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:gen_all_oplist) " +
               "--model_file_list_path $(@query_outputs \"filter('.*_et_oplist', deps(set({deps})))\") " +
               "--allow_include_all_overloads " +
               "--check_ops_not_overlapping " +
               "--DEBUG_ONLY_check_prim_ops $(@query_targets \"filter('prim_ops_registry(?:_static|_aten)?$', deps(set({deps})))\") " +
               "--output_dir $OUT ").format(deps = " ".join(["\'{}\'".format(d) for d in deps])),
        define_static_target = False,
        platforms = kwargs.pop("platforms", get_default_executorch_platforms()),
        outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
        default_outs = ["."],
        **kwargs,
    )

def check_recursive_dependencies(
    name,
    parent,
    child,
    message = "",
    **kwargs,
):
    """
    Checks if child is a transitive dependency of parent and fails if it is.
    The query runs the equivalent of `buck2 uquery "allpaths(parent, child)".
    The path from parent->child is available in the out file and error message.
    """
    message = "Dependency violation: '{}' should not depend on '{}'. {}".format(parent, child, message)

    if parent == child:
        fail(message)

    runtime.genrule(
        name = name,
        macros_only = False,
        cmd = 'mkdir -p $OUT;paths="$(query_targets allpaths({}, {}))"; echo "$paths" > $OUT/dep.txt; if [ -z "$paths" ]; then echo "Dependencies look good"; else echo {}. This will cause duplicate symbol errors when building with dtype selective build. The dependency path is: "$paths"; fail; fi'.format(parent, child, message),
        define_static_target = False,
        # The path is saved to $OUT/dep.txt and can be accessed via genrule_name[result].
        outs = {"result": ["dep.txt"]},
        default_outs = ["."],
        platforms = kwargs.pop("platforms", get_default_executorch_platforms()),
    )
