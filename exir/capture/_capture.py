# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch._export
from executorch.exir.capture._config import CaptureConfig
from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.program import ExirExportedProgram, MultiMethodExirExportedProgram
from executorch.exir.program._program import HackedUpExportedProgramDONOTUSE
from executorch.exir.tracer import (
    _default_decomposition_table,
    dispatch_trace,
    dynamo_trace,
    flatten_output,
    Value,
)
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.eval_frame import Constraint
from torch._export import CallSpec, export, ExportedProgram, ExportGraphSignature
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.func import functionalize
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils import _pytree as pytree


Val = Any


CompileSpec = namedtuple(
    "CompileSpec", ["method_name", "callable", "args", "constraints"]
)


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

    ep = HackedUpExportedProgramDONOTUSE(
        graph_module,
        graph_module.graph,
        ExportGraphSignature([], [], user_inputs, user_outputs, {}, {}, {}, None),
        CallSpec(in_spec, out_spec),
        {},
        {},
        [],
        [],
        None,
    )
    return ExirExportedProgram(ep, False)


@compatibility(is_backward_compatible=False)
def capture(  # noqa: C901
    f: Callable[..., Any],
    args: Tuple[Value, ...],
    config: Optional[CaptureConfig] = None,
    constraints: Optional[List[Constraint]] = None,
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

            ep = export(f, args, constraints=constraints)
            ep = ep.run_decompositions(_default_decomposition_table())  # pyre-ignore[6]
            ep = ep._transform(ReplaceViewOpsWithViewCopyOpsPass())
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
                constraints=constraints,
                _use_old_decomp_table=config._use_old_decomp_table,
            )

        else:
            graph_module, _ = dynamo_trace(
                f,
                args,
                aten_graph=True,
                tracing_mode="fake",
                dynamo_config=config._dynamo_config,
                constraints=None,  # constraints make sense only when dynamic shapes is enabled
                _use_old_decomp_table=config._use_old_decomp_table,
            )

        if out_spec is None:
            out_spec = (
                graph_module.graph._codegen.pytree_info.out_spec
                or pytree.tree_flatten(f(*args))[1]
            )

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
        node.name for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    output_node = list(graph_module.graph.nodes)[-1]
    assert output_node.op == "output"
    user_outputs = [arg.name for arg in output_node.args[0]]

    ep = ExportedProgram(
        graph_module,
        graph_module.graph,
        ExportGraphSignature([], [], user_inputs, user_outputs, {}, {}, {}, None),
        CallSpec(in_spec, out_spec),
        {},
        {},
        [],
        [],
        dialect="OLD_EXIR_ATEN",
    )
    return ExirExportedProgram(ep, False)


@compatibility(is_backward_compatible=False)
def capture_multiple(
    m: Union[torch.nn.Module, Callable[..., Any]],
    args: Union[Dict[str, Tuple[Value, ...]], Tuple[Value, ...]],
    config: Optional[CaptureConfig] = None,
    prim_getters: Optional[Set[str]] = None,
    constraints: Optional[Union[Dict[str, List[Constraint]], List[Constraint]]] = None,
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

        constraints: Input shape constraints.

        When `m` is an nn.Module, `constraints` is a dictionary that maps names of method
        to their input shape constraints.

        When `m` is a non-Module callable, `constraints` is a list of input shape constraints.

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
        "This function is now deprecated, please use `torch.export and exir.to_edge` instead. ",
        DeprecationWarning,
        stacklevel=1,
    )
    # Normalize m and args.
    compile_specs = []
    prim_getter_cache: Optional[Dict[str, Any]] = None
    if isinstance(m, torch.nn.Module):
        if constraints is not None:
            assert isinstance(
                constraints, dict
            ), f"Expected a dict for constraints, got {type(constraints)}"

        if isinstance(args, tuple):
            compile_specs.append(
                CompileSpec(
                    "forward",
                    m.forward,
                    args,
                    constraints["forward"]
                    if constraints and "forward" in constraints
                    else None,
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
                        constraints[method_name]
                        if constraints and method_name in constraints
                        else None,
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
        if constraints is not None:
            assert isinstance(
                constraints, list
            ), f"Expected a list for constraints, got {type(constraints)}"
        compile_specs.append(CompileSpec("forward", m, args, constraints))

    method_name_to_prog = {}
    for compile_spec in compile_specs:
        method_name_to_prog[compile_spec.method_name] = capture(
            compile_spec.callable, compile_spec.args, config, compile_spec.constraints
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
