import copy
import sys
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from packaging import version
from torch_tensorrt._utils import sanitized_torch_version

import torch

if version.parse(sanitized_torch_version()) >= version.parse("2.dev"):
    import torch._dynamo as torchdynamo

from torch.fx.passes.infra.pass_base import PassResult
from torch_tensorrt.fx.utils import req_torch_version
from torch_tensorrt.fx.passes.lower_basic_pass_aten import (
    compose_bmm,
    compose_chunk,
    compose_getitem_slice,
    remove_ops,
    replace_aten_op_with_indices,
    replace_aten_reshape_alias_with_replace,
    replace_builtin_ops,
    replace_inplace_ops,
    replace_native_layernorm_with_layernorm,
    replace_transpose_mm_op_with_linear,
    run_const_fold,
)
from typing_extensions import TypeAlias

Value: TypeAlias = Union[
    Tuple["Value", ...],
    List["Value"],
    Dict[str, "Value"],
]


class DynamoConfig:
    """
    Manage Exir-specific configurations of Dynamo.
    """

    def __init__(
        self,
        capture_scalar_outputs: bool = True,
        guard_nn_modules: bool = True,
        dynamic_shapes: bool = True,
        specialize_int: bool = True,
        verbose: bool = True,
    ) -> None:
        self.capture_scalar_outputs = capture_scalar_outputs
        self.guard_nn_modules = guard_nn_modules
        self.dynamic_shapes = dynamic_shapes
        self.specialize_int = specialize_int
        self.verbose = verbose

    def activate(self) -> None:
        torchdynamo.config.capture_scalar_outputs = self.capture_scalar_outputs
        torchdynamo.config.guard_nn_modules = self.guard_nn_modules
        torchdynamo.config.dynamic_shapes = self.dynamic_shapes
        torchdynamo.config.specialize_int = self.specialize_int
        torchdynamo.config.verbose = self.verbose

    def deactivate(self) -> None:
        torchdynamo.config.capture_scalar_outputs = True
        torchdynamo.config.guard_nn_modules = True
        torchdynamo.config.dynamic_shapes = True
        torchdynamo.config.specialize_int = True
        torchdynamo.config.verbose = True


@contextmanager
def using_config(config: DynamoConfig) -> Generator[DynamoConfig, None, None]:
    config.activate()
    try:
        yield config
    finally:
        config.deactivate()


@contextmanager
def setting_python_recursive_limit(limit: int = 10000) -> Generator[None, None, None]:
    """
    Temporarily increase the python interpreter stack recursion limit.
    This is mostly used for pickling large scale modules.
    """
    default = sys.getrecursionlimit()
    if limit > default:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(default)


@req_torch_version("2.dev")
def dynamo_trace(
    f: Callable[..., Value],
    # pyre-ignore
    args: Tuple[Any, ...],
    aten_graph: bool,
    tracing_mode: str = "real",
    dynamo_config: Optional[DynamoConfig] = None,
) -> Tuple[torch.fx.GraphModule, Set]:
    """
    TODO: Once we fully migrate to torchdynamo frontend, we will remove
    this config option alltogether.  For now, it helps with quick
    experiments with playing around with TorchDynamo
    """
    if dynamo_config is None:
        dynamo_config = DynamoConfig()
    with using_config(dynamo_config), setting_python_recursive_limit(2000):
        torchdynamo.reset()
        try:
            return torchdynamo.export(
                f,
                *copy.deepcopy(args),
                aten_graph=aten_graph,
                tracing_mode=tracing_mode,
            )
        except torchdynamo.exc.Unsupported as exc:
            raise RuntimeError(
                "The user code is using a feature we don't support. "
                "Please try torchdynamo.explain() to get possible the reasons",
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "torchdynamo internal error occured. Please see above stacktrace"
            ) from exc


@req_torch_version("2.dev")
def trace(f, args, *rest):
    graph_module, guards = dynamo_trace(f, args, True, "symbolic")
    return graph_module, guards


@req_torch_version("2.dev")
def opt_trace(f, args, *rest):
    """
    Optimized trace with necessary passes which re-compose some ops or replace some ops
    These passes should be general and functional purpose
    """
    passes_list = [
        compose_bmm,
        compose_chunk,
        compose_getitem_slice,
        replace_aten_reshape_alias_with_replace,
        replace_aten_op_with_indices,
        replace_transpose_mm_op_with_linear,  # after compose_bmm
        replace_native_layernorm_with_layernorm,
        remove_ops,
        replace_builtin_ops,  # after replace_native_layernorm_with_layernorm
        replace_inplace_ops,  # remove it once functionalization is enabled
    ]

    fx_module, _ = trace(f, args)
    print(fx_module.graph)
    for passes in passes_list:
        pr: PassResult = passes(fx_module)
        fx_module = pr.graph_module

    fx_module(*args)

    fx_module = run_const_fold(fx_module)
    print(fx_module.graph)
    return fx_module
