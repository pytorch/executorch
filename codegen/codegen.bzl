load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_xplat")
load("@fbsource//xplat/caffe2:pt_ops.bzl", "pt_operator_library")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_default_executorch_platforms", "runtime")

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
        exported_deps = [],
        model = None,
        include_all_operators = False,
        ops_schema_yaml_target = None,
        define_static_targets = False,
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
    if model:
        genrule_cmd.append(
            "--model_file_path=$(location {})".format(model),
        )
    if ops_schema_yaml_target or ops or model:
        runtime.genrule(
            name = name,
            macros_only = False,
            cmd = " ".join(genrule_cmd),
            out = "selected_operators.yaml",
            labels = ["pt_operator_library"],
            **kwargs
        )
    else:
        kwargs["exported_deps"] = exported_deps
        kwargs["include_all_operators"] = include_all_operators
        pt_operator_library(
            name = name,
            **kwargs
        )
        if define_static_targets:
            pt_operator_library(
                name = name + "_static",
                **kwargs
            )

def _get_headers(genrule_name, prefix = "", custom_op = None):
    return {
        prefix + f: ":{}[{}]".format(genrule_name, f)
        for f in OPERATOR_HEADERS + (CUSTOM_OPS_NATIVE_FUNCTION_HEADER if custom_op else [])
    }

def _prepare_genrule_and_lib(
        name,
        functions_yaml_path = None,
        custom_ops_yaml_path = None,
        custom_ops_aten_kernel_deps = [],
        custom_ops_requires_runtime_registration = True,
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
    genrule_cmd = [
        "$(exe fbsource//xplat/caffe2/torchgen:gen_executorch)",
        "--source-path=$(location //executorch/codegen:templates)",
        "--tags-path $(location fbsource//xplat/caffe2:aten_src_path)/aten/src/ATen/native/tags.yaml",
        "--aten_yaml_path $(location fbsource//xplat/caffe2:aten_src_path)/aten/src/ATen/native/native_functions.yaml",
        "--install_dir=${OUT}",
        # TODO(dbort): Add a second step that verifies that the set of
        # actually-generated files matches GENERATED_FILES.
    ]

    # The command will always generate these files.
    genrule_outs = GENERATED_SOURCES + OPERATOR_HEADERS + (CUSTOM_OPS_NATIVE_FUNCTION_HEADER if custom_ops_yaml_path else [])

    # Determine what sources custom_ops_<name> target should include
    custom_ops_sources = CUSTOM_OPS_SCHEMA_REGISTRATION_SOURCES + (
        CUSTOM_OPS_GENERATED_SOURCES if custom_ops_aten_kernel_deps else CUSTOM_OPS_DUMMY_KERNEL_SOURCES
    )

    genrules = {}
    libs = {}

    # if aten_mode is true, we don't need functions_yaml_path
    genrule_name = name + "_combined"
    headers = _get_headers(genrule_name = genrule_name, custom_op = custom_ops_yaml_path)

    # need to register ATen ops into Executorch runtime:
    need_reg_aten_ops = aten_mode or functions_yaml_path

    # need to register custom ops into Executorch runtime:
    need_reg_custom_ops = custom_ops_yaml_path and custom_ops_requires_runtime_registration

    need_reg_ops = need_reg_aten_ops or need_reg_custom_ops

    if need_reg_aten_ops:
        path = (
            "$(location fbsource//xplat/caffe2:aten_src_path)/aten/src/ATen/native/native_functions.yaml"
        ) if not functions_yaml_path else functions_yaml_path
        genrule_cmd = genrule_cmd + [
            "--functions_yaml_path={}".format(path),
        ]
    if aten_mode:
        genrule_cmd = genrule_cmd + ["--use_aten_lib"]
    if custom_ops_yaml_path:
        genrule_cmd = genrule_cmd + [
            "--custom_ops_yaml_path=" + custom_ops_yaml_path,
        ]
        genrule_outs += custom_ops_sources
    genrules[genrule_name] = {
        "cmd": genrule_cmd,
        "outs": genrule_outs,
    }

    if need_reg_ops:
        libs[name] = {
            "genrule": genrule_name,
            "headers": headers,
            "srcs": GENERATED_SOURCES,
        }

    header_lib = name + "_headers"
    libs[header_lib] = {
        "headers": headers,
    }
    if custom_ops_yaml_path:
        # lib for registering custom ops to pytorch
        libs["custom_ops_" + name] = {
            "genrule": genrule_name,
            "headers": headers,
            "srcs": custom_ops_sources,
        }
        if header_lib in libs:
            libs[header_lib]["headers"].update(headers)
        else:
            libs[header_lib] = {
                "headers": headers,
            }
    return genrules, libs

def executorch_generated_lib(
        name,
        functions_yaml_target = None,
        custom_ops_yaml_target = None,
        fallback_yaml_target = None,
        define_static_targets = False,
        custom_ops_aten_kernel_deps = [],
        custom_ops_requires_runtime_registration = True,
        visibility = [],
        aten_mode = False,
        use_default_aten_ops_lib = True,
        deps = [],
        xplat_deps = [],
        fbcode_deps = [],
        platforms = get_default_executorch_platforms(),
        compiler_flags = []):
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
        use_default_aten_ops_lib: If `aten_mode` is True AND this flag is True, use `torch_mobile_all_ops` for ATen operator library.
        xplat_deps: Additional xplat deps, can be used to provide custom operator library.
        fbcode_deps: Additional fbcode deps, can be used to provide custom operator library.
        compiler_flags: compiler_flags args to runtime.cxx_library
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
        custom_ops_aten_kernel_deps = custom_ops_aten_kernel_deps,
        custom_ops_requires_runtime_registration = custom_ops_requires_runtime_registration,
        aten_mode = aten_mode,
    )

    # genrule for selective build from static operator list
    oplist_dir_name = name + "_pt_oplist"
    runtime.genrule(
        name = oplist_dir_name,
        macros_only = False,
        cmd = ("$(exe fbsource//xplat/executorch/codegen/tools:gen_all_oplist) " +
               "--model_file_list_path $(@query_outputs 'attrfilter(labels, pt_operator_library, deps(set({deps})))') " +
               "--allow_include_all_overloads " +
               "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
        outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
        default_outs = ["."],
        platforms = platforms,
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
                "//executorch/kernels:kernel_runtime_context" + aten_suffix,
            ],
        )

    xplat_deps = xplat_deps + (["//xplat/caffe2:torch_mobile_all_ops"] if aten_mode and use_default_aten_ops_lib else [])
    fbcode_deps = fbcode_deps + (["//caffe2:libtorch"] if aten_mode and use_default_aten_ops_lib else [])
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
            exported_preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            # link_whole is necessary because the operators register themselves via
            # static initializers that run at program startup.
            # @lint-ignore BUCKLINT link_whole
            link_whole = True,
            visibility = visibility,
            # Operator Registration is done through static tables
            compiler_flags = ["-Wno-global-constructors"] + compiler_flags,
            deps = [
                "//executorch/core:operator_registry",
                "//executorch/core/prim_ops:prim_ops_registry" + aten_suffix,
                "//executorch/core/values:executor_values" + aten_suffix,
                "//executorch/profiler:profiler",
            ] + deps,
            exported_deps = [
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
                "//executorch/kernels:kernel_runtime_context" + aten_suffix,
            ],
            xplat_deps = xplat_deps,
            fbcode_deps = fbcode_deps,
            define_static_target = define_static_targets,
            # Relax visibility restrictions since deps may include targets outside
            # of //executorch.
            _is_external_target = True,
            platforms = platforms,
        )

    # If custom ops are provided, emit a host-only C++ library that declares and
    # registers them. Clients can load this library into local PyTorch using
    # `torch.ops.load_library()` to make them visible while authoring models.
    #
    # For the embedded runtime, clients should depend on the `<name>`
    # cxx_library above, which will register the custom ops as long as
    # custom_ops_requires_runtime_registration is True.
    compiler_lib = "custom_ops_" + name if "custom_ops_" + name in libs else None
    if compiler_lib:
        # The library needs to be able to include <torch/library.h>.
        if is_xplat():
            torch_dep = ["//xplat/caffe2:torch"]
        else:
            torch_dep = ["//caffe2:libtorch"]

        # TODO(T129125039): Rename this to make it clear that it's not part of
        # the embedded runtime; it's only for registering custom ops with the
        # PyTorch authoring runtime.
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
            deps = [
                "//executorch/core/kernel_types:kernel_types_aten",
                "//executorch/core:core",
            ] + torch_dep + custom_ops_aten_kernel_deps,
            exported_deps = [
                "//executorch/kernels:kernel_runtime_context_aten",
            ],
            define_static_target = define_static_targets,
            # Relax visibility restrictions since deps may include targets
            # outside of //executorch.
            _is_external_target = True,
            # Explicitly indicate that this C++ library will be loaded by Python
            # and consequently need to be exposed as shared libraries. It's not
            # required, but when set it'll make builds faster.
            supports_python_dlopen = True,
            platforms = platforms,
            compiler_flags = compiler_flags,
        )
