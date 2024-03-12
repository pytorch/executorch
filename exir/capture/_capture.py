# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from collections import namedtuple
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.exir.capture._config import CaptureConfig
from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.program import ExirExportedProgram, MultiMethodExirExportedProgram
from executorch.exir.program._program import _transform, HackedUpExportedProgramDONOTUSE
from executorch.exir.tracer import (
    _default_decomposition_table,
    dispatch_trace,
    dynamo_trace,
    flatten_output,
    Value,
)
from executorch.exir.verification.verifier import EXIRATenDialectVerifierBase
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import export
from torch.export.exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    ModuleCallEntry,
    ModuleCallSignature,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.func import functionalize
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils import _pytree as pytree


Val = Any


CompileSpec = namedtuple(
    "CompileSpec", ["method_name", "callable", "args", "dynamic_shapes"]
)


CallSpec = namedtuple("CallSpec", ["in_spec", "out_spec"])


@compatibility(is_backward_compatible=False)
def _capture_legacy_do_not_use(f, args) -> ExirExportedProgram:
    """
    This is a legacy API that should be avoided. Prefer to use capture() instead.
    """
    warnings.warn(
        "This function is now deprecated, please use `torch.export and exir.to_edge` instead. "
        "See https://github.com/pytorch/functorch for more details.",
        DeprecationWarning,
        stacklevel=1,
    )

    graph_module = dispatch_trace(f, args)
    flat_args = tuple(pytree.tree_flatten(args)[0])
    in_spec, out_spec = graph_module.in_spec, graph_module.out_spec

    _instantiate_missing_placeholder_val_with_real_inputs(graph_module, flat_args)
    graph_module._apply(torch.Tensor.contiguous)

    user_inputs = [
        node.name for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    output_node = list(graph_module.graph.nodes)[-1]
    assert output_node.op == "output"
    user_outputs = [arg.name for arg in output_node.args[0]]

    for n in graph_module.graph.nodes:
        if n.op == "call_function" and "val" not in n.meta:
            try:
                args, kwargs = pytree.tree_map_only(
                    torch.fx.Node, lambda x: x.meta["val"], (n.args, n.kwargs)
                )
                n.meta["val"] = n.target(*args, **kwargs)
            except Exception:
                n.meta["val"] = None

    ep = HackedUpExportedProgramDONOTUSE(
        root=graph_module,
        graph=graph_module.graph,
        graph_signature=ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=InputKind.USER_INPUT, arg=TensorArgument(name=i), target=None
                )
                for i in user_inputs
            ],
            output_specs=[
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT, arg=TensorArgument(name=o), target=None
                )
                for o in user_outputs
            ],
        ),
        call_spec=CallSpec(in_spec, out_spec),
        state_dict={},
        range_constraints={},
        module_call_graph=[
            ModuleCallEntry(
                fqn="",
                signature=ModuleCallSignature(
                    inputs=[],
                    outputs=[],
                    in_spec=in_spec,
                    out_spec=out_spec,
                ),
            )
        ],
        example_inputs=None,
        verifier=EXIRATenDialectVerifierBase,
    )
    return ExirExportedProgram(ep, False)


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.

    TODO(suo): upstream this to torch.export.wrapper.
    """
    # Save the original method
    original_method = obj.forward

    # Patch the method
    obj.forward = new_method.__get__(obj, obj.__class__)

    try:
        yield
    finally:
        # Restore the original method
        obj.forward = original_method


class WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.forward = f


@compatibility(is_backward_compatible=False)
def capture(  # noqa: C901
    f: Callable[..., Any],
    args: Tuple[Value, ...],
    config: Optional[CaptureConfig] = None,
    dynamic_shapes: Optional[List[Any]] = None,
) -> ExirExportedProgram:
    warnings.warn(
        "This function is now deprecated, please use `torch.export and exir.to_edge` instead. ",
        DeprecationWarning,
        stacklevel=1,
    )
    if not isinstance(args, tuple):
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            f"Expect `args` to be a tuple, got type: {type(args)}.",
        )

    config = config or CaptureConfig()
    out_spec = None
    # TODO (zhxchen17) Always functionalize in a second pass no matter which path is taken.
    flat_args = tuple(pytree.tree_flatten(args)[0])
    if not config.enable_aot:
        if config._unlift:
            raise ExportError(
                ExportErrorType.NOT_SUPPORTED,
                "_unlift config doesn't do anything without enable_aot enabled. Please do not set it",
            )
    if config.pt2_mode:
        if config.enable_aot:
            if config.enable_dynamic_shape:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    "Under enable_aot, enable_dynamic_shapes flag doesn't do anything. Please do not set it",
                )
            if not config.enable_functionalization:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    "Functionalization is required for enable_aot.",
                )

            # If trying to capture a method and the bound class instance is a
            # Module, then export the module while patching in that method.
            if isinstance(f, MethodType) and isinstance(f.__self__, torch.nn.Module):
                with patch_forward(f.__self__, f):
                    ep = export(
                        cast(torch.nn.Module, f.__self__),
                        args,
                        dynamic_shapes=dynamic_shapes,
                    )
            else:
                mod = f if isinstance(f, torch.nn.Module) else WrapperModule(f)
                ep = export(mod, args, dynamic_shapes=dynamic_shapes)

            ep = ep.run_decompositions(_default_decomposition_table())
            ep = _transform(ep, ReplaceViewOpsWithViewCopyOpsPass())
            if not config._unlift:
                return ExirExportedProgram(ep, False)
            graph_module = ep.module()

        elif config.enable_dynamic_shape:
            graph_module, _ = dynamo_trace(
                f,
                args,
                aten_graph=True,
                tracing_mode="symbolic",
                dynamo_config=config._dynamo_config,
                dynamic_shapes=dynamic_shapes,
                _use_old_decomp_table=config._use_old_decomp_table,
            )

        else:
            graph_module, _ = dynamo_trace(
                f,
                args,
                aten_graph=True,
                tracing_mode="fake",
                dynamo_config=config._dynamo_config,
                dynamic_shapes=None,
                _use_old_decomp_table=config._use_old_decomp_table,
            )

        if out_spec is None:
            if isinstance(graph_module.graph._codegen, torch.fx.graph._PyTreeCodeGen):
                out_spec = graph_module.graph._codegen.pytree_info.out_spec
            elif hasattr(graph_module, "_out_spec"):
                out_spec = graph_module._out_spec
            else:
                out_spec = pytree.tree_flatten(f(*args))[1]

        # NOTE (tmanlaibaatar)
        # torchdynamo.export adds extra kwarg into the graph module
        # which is then lost while we are calling make_fx. This is because
        # make_fx doesn't handle kwargs. Originally we used to use torchdynamo
        # input spec, but due to some limitations in pytree implementation, it doesn't
        # recognize the make_fx graph with torchdynamo input spec. We workaround it
        # by getting the input spec directly from user argument.
        in_spec = pytree.tree_flatten((args, {}))[1]

        if config.enable_functionalization and not config.enable_aot:
            args = copy.deepcopy(args)

            def graph_with_interpreter(*args):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph_module).run(*args)

            functionalized_callable = functionalize(
                graph_with_interpreter,
                remove="mutations_and_views",
            )
            assert isinstance(functionalized_callable, Callable)

            if config.enable_dynamic_shape:
                fake_tensor_mode = FakeTensorMode(
                    allow_fallback_kernels=False,
                    allow_non_fake_inputs=True,
                    shape_env=ShapeEnv(),
                )

                inps: List[torch.Tensor] = []
                for node in graph_module.graph.nodes:
                    if node.op == "placeholder" and "val" in node.meta:
                        example_fake_tensor = node.meta["val"]
                        assert isinstance(example_fake_tensor, FakeTensor)
                        inps.append(example_fake_tensor)

                if detected_fake_mode := _guards.detect_fake_mode(inps):
                    fake_tensor_mode = detected_fake_mode

                count = 0

                def convert_to_fake(x):
                    nonlocal count
                    val = inps[count]
                    count += 1
                    return val

                fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)

                with enable_python_dispatcher(), fake_tensor_mode:
                    graph_module = make_fx(
                        functionalized_callable,
                        tracing_mode="real",
                        _allow_non_fake_inputs=True,
                    )(*fake_args)
            else:
                # To avoid breaking folks, use the deprecated "real" tracing
                # mode if we're not using pt2.
                tracing_mode = "fake" if config.pt2_mode else "real"
                graph_module = make_fx(
                    functionalized_callable,
                    tracing_mode=tracing_mode,
                    _allow_non_fake_inputs=True,
                )(*args)

        flatten_output(graph_module)

    else:
        raise InternalError("pt2=False path is officially deprecated")

    _instantiate_missing_placeholder_val_with_real_inputs(graph_module, flat_args)
    graph_module._apply(torch.Tensor.contiguous)

    user_inputs = [
        InputSpec(
            kind=InputKind.USER_INPUT, arg=TensorArgument(name=node.name), target=None
        )
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    output_node = list(graph_module.graph.nodes)[-1]
    assert output_node.op == "output"
    user_outputs = [
        OutputSpec(
            kind=OutputKind.USER_OUTPUT, arg=TensorArgument(name=arg.name), target=None
        )
        for arg in output_node.args[0]
    ]

    graph_module.graph.eliminate_dead_code()
    ep = ExportedProgram(
        root=graph_module,
        graph=graph_module.graph,
        graph_signature=ExportGraphSignature(user_inputs, user_outputs),
        state_dict={},
        range_constraints={},
        module_call_graph=[
            ModuleCallEntry(
                fqn="",
                signature=ModuleCallSignature(
                    inputs=[],
                    outputs=[],
                    in_spec=in_spec,
                    out_spec=out_spec,
                ),
            )
        ],
        example_inputs=None,
        verifier=EXIRATenDialectVerifierBase,
    )
    return ExirExportedProgram(ep, False)


@compatibility(is_backward_compatible=False)
def capture_multiple(
    m: Union[torch.nn.Module, Callable[..., Any]],
    args: Union[Dict[str, Tuple[Value, ...]], Tuple[Value, ...]],
    config: Optional[CaptureConfig] = None,
    prim_getters: Optional[Set[str]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], List[Any]]] = None,
):
    """
    capture_multiple traces either an nn.Module or just a callable with PyTorch
    operations inside and produce a single MultiMethodExirExportedProgram that
    can potentially have multiple entry points. When multiple entry points
    are traced, each of them is stored separately in the resulting
    MultiMethodExirExportedProgram without sharing state.

    Args:
        m: the `nn.Module` or callable to trace.

        args: Tracing example inputs.

        When `m` is an nn.Module, `args` can be
        1) A dictionary that maps names of method to their tracing example inputs.
        in this case, all specified methods will be captured.
        2) A tuple. In this case, `forward` method of `m` will be captured. It is
        equivalent to passing {"forward", tuple-type-args}

        When `m` is a non-Module callable, `args` must be a Tuple containing
        tracing example inputs.

        config: A CaptureConfig object that specifies how to interpret the
        program being captured.

        prim_getters: A set of primitive getter functions to capture the return values of

        dynamic_shapes: Input dynamic shapes.

        When `m` is an nn.Module, `dynamic_shapes` is a dictionary that maps names of method
        to their input dynamic shapes.

        When `m` is a non-Module callable, `dynamic_shapes` is a list of input dynamic shapes.

    Returns:
        A MultiMethodExirExportedProgram.

        if `m` is an nn.Module, returned program would have multiple
        captured methods, each corresponding to one entry in args dictionary.

        if `m` is a non-Module callable, returned program would have a single
        captured method named `forward`.

    Raises:
        AssertionError if given method name do not reference a valid method
        on the given nn.Module.
    """
    warnings.warn(
        "This function is now deprecated, please use `torch.export and exir.to_edge` instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    # Normalize m and args.
    compile_specs = []
    prim_getter_cache: Optional[Dict[str, Any]] = None
    if isinstance(m, torch.nn.Module):
        if dynamic_shapes is not None:
            assert isinstance(
                dynamic_shapes, dict
            ), f"Expected a dict for dynamic_shapes, got {type(dynamic_shapes)}"

        if isinstance(args, tuple):
            compile_specs.append(
                CompileSpec(
                    "forward",
                    m.forward,
                    args,
                    (
                        dynamic_shapes["forward"]
                        if dynamic_shapes and "forward" in dynamic_shapes
                        else None
                    ),
                )
            )
        else:
            assert isinstance(
                args, dict
            ), f"Expected a tuple or Dict[str, tuple], got {type(args)}"
            for method_name, method_args in args.items():
                compile_specs.append(
                    CompileSpec(
                        method_name,
                        getattr(m, method_name),
                        method_args,
                        (
                            dynamic_shapes[method_name]
                            if dynamic_shapes and method_name in dynamic_shapes
                            else None
                        ),
                    )
                )
        if prim_getters is not None:
            prim_getter_cache = {}
            for getter in prim_getters:
                prim_getter_cache[getter] = getattr(m, getter)()
    else:
        # Reaching here means `m` is a non-Module callable.
        assert isinstance(
            m, Callable
        ), f"Only nn.Module or callable allowed, got {type(m)}"
        assert isinstance(
            args, tuple
        ), f"When tracing a non-Module callable, `args` must be a tuple of tracing inputs, but got {type(args)}"
        assert (
            prim_getters is None
        ), "Caller should not specify primitive getter functions when only providing a callable as input"
        if dynamic_shapes is not None:
            assert isinstance(
                dynamic_shapes, list
            ), f"Expected a list for constraints, got {type(dynamic_shapes)}"
        compile_specs.append(CompileSpec("forward", m, args, dynamic_shapes))

    method_name_to_prog = {}
    for compile_spec in compile_specs:
        method_name_to_prog[compile_spec.method_name] = capture(
            compile_spec.callable,
            compile_spec.args,
            config,
            compile_spec.dynamic_shapes,
        )

    return MultiMethodExirExportedProgram(method_name_to_prog, prim_getter_cache)


# This is to bootstrap the missing meta["val"] when 1. ph consists of scalar
# 2. meta["val"] is not properly set in dispatch_trace.
def _instantiate_missing_placeholder_val_with_real_inputs(gm, args):
    phs = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(phs) != len(args):
        raise ExportError(
            ExportErrorType.NOT_SUPPORTED,
            "Expect number of placeholders to be the same as user inputs.",
        )
    for node, arg in zip(phs, args):
        if "val" not in node.meta or node.meta["val"] is None:
            node.meta["val"] = arg
