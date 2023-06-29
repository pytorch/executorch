# pyre-strict

import types
from typing import List, Tuple

import torch
import torch.utils._pytree as pytree
from executorch.backends.compile_spec_schema import CompileSpec
from executorch.exir.graph_module import (
    _get_submodule,
    EXIR_METADATA,
    make_export_graph_module,
)
from executorch.exir.tracer import Value
from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_slot,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.passes.utils.fuser_utils import (
    erase_nodes,
    fuse_as_graphmodule,
    insert_subgm,
    legalize_graph,
    NodeList,
    topo_sort,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)

from torch.utils._pytree import tree_flatten


class LoweredBackendModule(torch.nn.Module):
    """
    A subclass of nn.Module that is generated for modules containing
    delegated functions. This is can be created by calling `to_backend`.

    Private Attributes:
        * **backend_id**: The backend's name
        * **processed_bytes**: The delegate blobs created from backend.preprocess
        * **compile_specs**: A list of backend-specific objects with static
            metadata to configure the "compilation" process.
        * **original_module**: The original EXIR module
    """

    _backend_id: str
    _processed_bytes: bytes
    _compile_specs: List[CompileSpec]
    _original_module: torch.fx.GraphModule

    def __init__(
        self,
        edge_ir_m: torch.fx.GraphModule,
        backend_id: str,
        processed_bytes: bytes,
        compile_specs: List[CompileSpec],
    ) -> None:
        super().__init__()
        self._original_module = edge_ir_m
        self._backend_id = backend_id
        self._processed_bytes = processed_bytes
        self._compile_specs = compile_specs

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def processed_bytes(self) -> bytes:
        return self._processed_bytes

    @property
    def compile_specs(self) -> List[CompileSpec]:
        return self._compile_specs

    @property
    def original_module(self) -> torch.fx.GraphModule:
        return self._original_module

    # Used to patch each delegated function with a call_delegate call
    @staticmethod
    def patched_method(
        backend_module: "LoweredBackendModule",
        *args: Value,
        **kwargs: Tuple[Value, ...],
    ) -> Value:
        return executorch_call_delegate(backend_module, *args)


executorch_call_delegate = HigherOrderOperator(
    "executorch_call_delegate", _deprecated_global_ns=True
)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.BackendSelect)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)


# pyre-ignore
def trace_call_delegate(proxy_mode, func_overload, lowered_module, *args):
    # pyre-ignore
    def _unwrap_proxy(e):
        if not isinstance(e, (torch.Tensor, torch.SymInt, torch.SymFloat)):
            return e
        return get_proxy_slot(e, proxy_mode.tracer, e, lambda e: e.proxy)

    if not isinstance(lowered_module, LoweredBackendModule):
        raise ValueError(
            "executorch_call_delegate()'s first argument must be a LoweredBackendModule"
        )

    with disable_proxy_modes_tracing():
        out = lowered_module.original_module(*args)

    lowered_name = get_lowered_module_name(proxy_mode.tracer.root, lowered_module)
    proxy_mode.tracer.root.register_module(lowered_name, lowered_module)

    node_args = (lowered_module, *args)
    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="executorch_call_delegate"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def call_delegate_cpu(lowered_module, *args):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU key"
    return lowered_module.original_module(*args)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.Autograd)
# pyre-ignore
def call_delegate_autograd(lowered_module, *args):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([lowered_module, *args])
    requires_grad = any(
        [f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)]
    )

    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
    ):
        res = executorch_call_delegate(lowered_module, *args)

        if requires_grad:
            err_fn = torch._C._functions.DelayedError(
                b"NYI: call_delegate doesn't support autograd",
                1,
            )
            # Create aliases of the output that has requires_grad=True. We need
            # at least one of the inputs to err_fn to require grad so that the
            # output will have a grad_fn.

            # pyre-ignore
            def fake_requires_grad(var):
                if var is not None:
                    var = var.detach()
                    var.requires_grad = True
                return err_fn(var)

            return pytree.tree_map(fake_requires_grad, res)

        return res


@executorch_call_delegate.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def call_delegate_proxy_torch_dispatch_mode(lowered_module, *args):
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_call_delegate(mode, executorch_call_delegate, lowered_module, *args)
    return res


@executorch_call_delegate.py_impl(FakeTensorMode)
# pyre-ignore
def call_delegate_fake_tensor_mode(lowered_module, *args):
    return lowered_module.original_module(*args)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.Functionalize)
# pyre-ignore
def call_delegate_func(lowered_module, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_args = tuple(
        _unwrap_all_tensors_from_functional(arg, reapply_views=reapply_views)
        for arg in args
    )
    guard = torch._C.ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    )
    try:
        delegate_return = executorch_call_delegate(lowered_module, *unwrapped_args)
        return _wrap_all_tensors_to_functional(delegate_return, level=0)
    finally:
        del guard


# pyre-ignore
@executorch_call_delegate.py_impl(torch._C._functorch.TransformType.Functionalize)
# pyre-ignore
def call_delegate_functionalize(interpreter, lowered_module, *args):
    """
    Functionalization implementation for torch.ops.executorch_call_delegate. We
    don't need to do anything since the delegated program is controlled by
    users.
    """
    reapply_views = interpreter.functionalize_add_back_views()
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_args = tuple(
        _unwrap_all_tensors_from_functional(arg, reapply_views=reapply_views)
        for arg in args
    )

    with interpreter.lower():
        res = executorch_call_delegate(lowered_module, *unwrapped_args)
        return _wrap_all_tensors_to_functional(res, level=interpreter.level())


def patch_lowered_functions(module: LoweredBackendModule) -> None:
    """
    Patches the forward function that will be delegated so that during tracing,
    all callsites to the graph module will instead have a
    "executorch_call_delegate" op in the FX graph.

    Args:
        module: A module that should contain the attributes contained in a
            lowered module (``compile_specs``,
            ``processed_bytes``, ``backend_id``, and
            ``original_module``)

    Returns:
        A module where if called during tracing, we will insert a
        executorch_call_delegate op into the FX graph marking a callsite to
        these delegated functions.
    """
    if not isinstance(module, LoweredBackendModule):
        return

    # Monkey-patch the forward function
    # pyre-ignore
    module.forward = types.MethodType(LoweredBackendModule.patched_method, module)


def get_lowered_module_name(
    root: torch.nn.Module, lowered_module: LoweredBackendModule
) -> str:
    """
    Adds the given lowered_module into the given root module and returns the
    name of the module added.
    """
    # Find a qualifying name for the lowered submodule
    qualname = None
    i = 0
    while True:
        qualname = f"lowered_module_{i}"
        if not hasattr(root, qualname):
            break
        i += 1
    assert qualname is not None

    root.add_module(qualname, lowered_module)
    return qualname


# TODO(zhxchen17) Try ExportPass
def _fixup_output_node(gm: torch.fx.GraphModule) -> None:
    for node in reversed(gm.graph.nodes):
        if node.op == "output":
            with gm.graph.inserting_before(node):
                assert len(node.args) == 1
                outputs = node.args[0]
                if isinstance(outputs, torch.fx.Node):
                    val = outputs.meta.get("val")
                    if isinstance(val, list):
                        # If a list is returned, in some cases it is represented as a
                        # singular node, like `split_copy_tensor` but EXIR will return a
                        # opened-up list like `[getitem1, getitem2]`
                        outputs = [
                            torch.fx.Proxy(outputs)[i].node for i in range(len(val))
                        ]
            returns, out_spec = pytree.tree_flatten(outputs)
            node.args = (returns,)
            return


def generate_in_spec_out_spec(gm: torch.fx.GraphModule) -> None:
    output_nodes = []
    for node in gm.graph.nodes:
        if node.op == "output":
            output_nodes = node.args[0]

    all_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    (_, pytree_in) = tree_flatten(tuple(all_placeholders))
    (_, pytree_out) = tree_flatten(tuple(output_nodes))

    meta = gm.meta[EXIR_METADATA]
    meta.in_spec = pytree_in
    meta.out_spec = pytree_out


def create_submodule_from_nodes(
    gm: torch.fx.GraphModule,
    node_list: NodeList,
    tag: str,
    skip_legalize_graph: bool = False,
) -> Tuple[torch.fx.GraphModule, torch.fx.Node]:
    """
    Modifies the given graph module in-place to separate out the given nodes
    into a submodule. The given node_list should form a fully connected
    subgraph.

    Args:
        gm: The graph module that we want to partition
        node_list: A list of nodes that belong in the partition

    Returns:
        The submodule that has been partitioned, the call_module node in the
        toplevel graph module calling the submodule
    """
    sorted_nodes = topo_sort(node_list)

    submodule_name = "fused_" + tag
    initial_sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
        gm, sorted_nodes, submodule_name
    )

    _fixup_output_node(initial_sub_gm)
    sub_gm = make_export_graph_module(
        initial_sub_gm, initial_sub_gm.graph, submodule_name
    )

    gm = insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)
    if len(orig_outputs) == 1 and isinstance(orig_outputs[0].meta["val"], FakeTensor):
        # If the original output is a single tensor, it has been
        # pytree.tree_flatten-ed to be a singleton list, so we want to replace
        # all uses with a getitem call to the 0th index of the result
        for node in gm.graph.nodes:
            if node.op == "call_module":
                with gm.graph.inserting_after(node):
                    proxy_out = torch.fx.Proxy(node)[0].node  # type: ignore[index]
                    node.replace_all_uses_with(proxy_out, propagate_meta=True)
                    # Reset the args since it was overwritten in the previous line
                    proxy_out.args = (node, 0)

    erase_nodes(gm, sorted_nodes)

    # Topological sort original gm with newly created sub_gm
    # TODO : T153794167 Get rid of support for skipping legalize graph in create_submodule_from_nodes
    # once we transition to using fuse_by_partitions.
    if not skip_legalize_graph:
        legalize_graph(gm)

    # Get the call_module node
    submodule_node = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == submodule_name:
            submodule_node = node
        elif node.op == "call_module":
            raise RuntimeError(
                f"The submodule created with nodes {node_list} did not form \
                one fully contained subgraph. Check that these nodes form a \
                fully contained graph. Partitioned graph: {gm.graph}."
            )

    assert (
        submodule_node is not None
    ), f"No submodule was created with the nodes {node_list} in the graph {gm.graph}"

    generate_in_spec_out_spec(sub_gm)
    return sub_gm, submodule_node


def get_lowered_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, LoweredBackendModule, torch.fx.Node]]:
    """
    Returns a list of lowered modules that are in the given graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the lowered module that's stored in the graph module, the
    lowered module itself, and the fx node that called this lowered module).
    """
    lowered_submodules = []
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == executorch_call_delegate:
            name, module, node = _get_submodule(graph_module, node, 0)
            assert isinstance(module, LoweredBackendModule)
            lowered_submodules.append((name, module, node))
    return lowered_submodules
