load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_default_executorch_platforms", "is_xplat", "runtime", "struct_to_json")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "portable_header_list", "portable_source_list")

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

# Hide the dependency to caffe2 internally.
def et_operator_library(
        name,
        ops = [],
        ops_dict = {},
        model = None,
        include_all_operators = False,
        ops_schema_yaml_target = None,
        **kwargs):
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
        aten_mode = False):
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
    target = runtime.external_dep_location("gen-executorch")
    aten_src_path = runtime.external_dep_location("aten-src-path")
    genrule_cmd = [
        "$(exe {})".format(target),
        "--source-path=$(location //executorch/codegen:templates)",
        "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
        "--aten_yaml_path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
        "--install_dir=${OUT}",
        # TODO(dbort): Add a second step that verifies that the set of
        # actually-generated files matches GENERATED_FILES.
    ]

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
        deps = [],
        kernels = []):
    """Similar to _prepare_genrule_and_lib but for custom ops."""
    genrules = {}
    libs = {}
    aten_src_path = runtime.external_dep_location("aten-src-path")
    target = runtime.external_dep_location("gen-executorch")
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
            "$(exe {})".format(target),
            "--source-path=$(location //executorch/codegen:templates)",
            "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
            "--aten_yaml_path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
            "--custom_ops_yaml_path=" + custom_ops_yaml_path,
            "--install_dir=${OUT}",
            "--op_selection_yaml_path=$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
        ]

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
        genrules[genrule_name] = {
            "cmd": " ".join(genrule_cmd),
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
    """
    genrules, libs = _prepare_custom_ops_genrule_and_lib(
        name = name,
        custom_ops_yaml_path = "$(location {})".format(yaml_target),
        kernels = kernels,
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

# Used for dtype selective build. Genrules to copy source and header files.
def portable_outs(target_name, file_list):
    outs = {}
    for file in file_list:
        outs[file] = ["{}/{}".format(target_name, file)]
    return outs

def copy_portable_source_files(name):
    target_name = "portable_source_files"
    runtime.genrule(
        name = name,
        cmd = "cp -f -r $(location //executorch/kernels/portable/cpu:{}) $OUT/".format(target_name),
        outs = portable_outs(target_name, portable_source_list()),
        default_outs = ["."],
    )

def copy_portable_header_files(name):
    target_name = "portable_header_files"
    runtime.genrule(
        name = name,
        cmd = "cp -f -r $(location //executorch/kernels/portable/cpu:{}) $OUT/".format(target_name),
        outs = portable_outs(target_name, portable_header_list()),
        default_outs = ["."],
    )

def build_portable_lib(name, oplist_header_name, feature = None):
    """Build portable lib from source. We build from source so that the generated header file, 
    selected_op_variants.h, can be used to selectively build the lib for different dtypes.
    """

    # Copy portable cpp files.
    portable_source_files = []
    copy_portable_source_files_genrule = name + "_copy_portable_source"
    copy_portable_source_files(copy_portable_source_files_genrule)
    for op in portable_source_list():
        portable_source_files.append(":{}[{}]".format(copy_portable_source_files_genrule, op))

    # Copy portable header files.
    portable_header_files = {}
    copy_portable_header_files_genrule = name + "_copy_portable_header"
    copy_portable_header_files(copy_portable_header_files_genrule)
    for header in portable_header_list():
        portable_header_files[header] = ":{}[{}]".format(copy_portable_header_files_genrule, header)

    # Include dtype header.
    portable_header_files["selected_op_variants.h"] = ":{}[selected_op_variants]".format(oplist_header_name)

    # Build portable lib.
    runtime.cxx_library(
        name = name,
        srcs = portable_source_files,
        exported_headers = portable_header_files,
        exported_preprocessor_flags = ["-DEXECUTORCH_SELECTIVE_BUILD_DTYPE"],
        deps = ["//executorch/kernels/portable/cpu/pattern:all_deps", "//executorch/kernels/portable/cpu/util:all_deps"],
        # header_namespace is only available in xplat. See https://fburl.com/code/we2gvopk
        header_namespace = "executorch/kernels/portable/cpu",
        compiler_flags = ["-Wno-missing-prototypes"] +
                         # For shared library build, we don't want to expose symbols of
                         # kernel implementation (ex torch::executor::native::tanh_out)
                         # to library users. They should use kernels through registry only.
                         # With visibility=hidden, linker won't expose kernel impl symbols
                         # so it can prune unregistered kernels.
                         # Currently fbcode links all dependent libraries through shared
                         # library, and it blocks users like unit tests to use kernel
                         # implementation directly. So we enable this for xplat only.
                         ["-fvisibility=hidden"],
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
        feature = None):
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
        use_default_aten_ops_lib: If `aten_mode` is True AND this flag is True, use `torch_mobile_all_ops_et` for ATen operator library.
        xplat_deps: Additional xplat deps, can be used to provide custom operator library.
        fbcode_deps: Additional fbcode deps, can be used to provide custom operator library.
        compiler_flags: compiler_flags args to runtime.cxx_library
        dtype_selective_build: In additional to operator selection, dtype selective build further selects the dtypes for each operator. Can be used with model or dict selective build APIs, where dtypes can be specified. Note: this is only available in xplat.
        feature: Product-Feature Hierarchy (PFH). For internal use only, required for FoA in production. See: https://fburl.com/wiki/2wzjpyqy
    """
    if functions_yaml_target and aten_mode:
        fail("{} is providing functions_yaml_target in ATen mode, it will be ignored. `native_functions.yaml` will be the source of truth.".format(name))

    if not aten_mode and not functions_yaml_target and not custom_ops_yaml_target:
        fail("At least one of functions_yaml_target, custom_ops_yaml_target needs to be provided")

    aten_suffix = "_aten" if aten_mode else ""

    # merge functions.yaml with fallback yaml
    if functions_yaml_target:
        merge_yaml_name = name + "_merge_yaml"
        cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:merge_yaml) " +
               "--functions_yaml_path=$(location {}) ".format(functions_yaml_target) +
               "--output_dir=$OUT ")
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
        custom_ops_yaml_path = "$(location {})".format(custom_ops_yaml_target)
    else:
        custom_ops_yaml_path = None

    genrules, libs = _prepare_genrule_and_lib(
        name = name,
        functions_yaml_path = functions_yaml_path,
        custom_ops_yaml_path = custom_ops_yaml_path,
        custom_ops_requires_runtime_registration = custom_ops_requires_runtime_registration,
        aten_mode = aten_mode,
        manual_registration = manual_registration,
    )

    # genrule for selective build from static operator list
    oplist_dir_name = name + "_pt_oplist"
    runtime.genrule(
        name = oplist_dir_name,
        macros_only = False,
        cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:gen_all_oplist) " +
               "--model_file_list_path $(@query_outputs 'attrfilter(labels, et_operator_library, deps(set({deps})))') " +
               "--allow_include_all_overloads " +
               "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
        outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
        default_outs = ["."],
        platforms = platforms,
    )

    # genrule to generate selected_op_variants.h from selected_operators.yaml above
    oplist_header_name = name + "_et_op_dtype_gen"
    runtime.genrule(
        name = oplist_header_name,
        macros_only = False,
        cmd = ("$(exe //executorch/codegen/tools:gen_selected_op_variants) " +
               "--yaml_file_path $(location :{}[selected_operators.yaml]) " +
               "--output_dir $OUT").format(oplist_dir_name),
        outs = {"selected_op_variants": ["selected_op_variants.h"]},
        default_outs = ["."],
        platforms = platforms,
        visibility = visibility,
        _is_external_target = True,
    )

    # codegen genrule(s). For ATen mode we expect two genrules, one for ATen ops one for custom ops.
    for genrule_name in genrules:
        genrules[genrule_name]["cmd"].append(
            "--op_selection_yaml_path=$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
        )
        runtime.genrule(
            name = genrule_name,
            cmd = " ".join(genrules[genrule_name]["cmd"]),
            outs = {f: [f] for f in genrules[genrule_name]["outs"]},
            default_outs = ["."],
            platforms = platforms,
        )

    portable_lib = []
    if dtype_selective_build and is_xplat() and "//executorch/kernels/portable:operators" in kernel_deps:
        # Remove portable from kernel_deps as we're building it from source.
        kernel_deps.remove("//executorch/kernels/portable:operators")

        # Build portable lib.
        portable_lib_name = name + "_portable_lib"
        build_portable_lib(portable_lib_name, oplist_header_name, feature)
        portable_lib = [":{}".format(portable_lib_name)]

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
            compiler_flags = ["-Wno-global-constructors"] + compiler_flags,
            deps = [
                "//executorch/runtime/kernel:operator_registry",
                "//executorch/kernels/prim_ops:prim_ops_registry" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/codegen:macros",
            ] + deps + kernel_deps + portable_lib,
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
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
