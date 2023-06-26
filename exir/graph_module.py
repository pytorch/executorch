# pyre-strict

import copy
import dataclasses
import pickle
import warnings
from types import FunctionType as function
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from executorch.exir.common import setting_python_recursive_limit
from executorch.exir.error import InternalError
from executorch.exir.graph import ExportGraph

ExportGraphModule = fx.GraphModule


LeafValue = Union[
    torch.Tensor,
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.device,
    torch.memory_format,
    torch.layout,
    None,
]

# We maintain a global cache of op lookups as this significantly speeds up
# deserialization because hasattr(torch.ops, name) is an expensive call.
_cache_ops_dict: Dict[
    Tuple[str, str], Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]
] = {}
_cache_fake_ops_dict: Dict[Tuple[str, str], function] = {}


def reduce_graph_module(state_bytes: bytes) -> "ExportGraphModule":
    """
    Function used to deserialize a graph module.
    To serialize the graph, we mapped all of the targets within nodes to their
    string names since we cannot serialize the operations themselves. During
    deserialization, we will then replace the string target names with their
    actual operations.

    Args:
        body: Dictionary of properties for a graph module

    Returns:
        A loaded ExportGraphModule.
    """
    # pyre-ignore
    def str_to_op(str_op: str):
        if not isinstance(str_op, str):
            return str_op

        # Some source_fn values are just a string
        if not str_op.startswith("torch.ops."):
            return str_op

        # Get the torch op
        target = torch.ops
        for name in str_op.split(".")[2:]:
            if _cache_fake_ops_dict.get((str(target), name)):
                return _cache_fake_ops_dict[(str(target), name)]
            if _cache_ops_dict.get((str(target), name)):
                target = _cache_ops_dict[(str(target), name)]
                continue

            if not hasattr(target, name):
                warnings.warn(
                    f"Could not find operator {str_op}. Returning target as string."
                )
                # pyre-ignore
                def fake_op(x):
                    return x

                fake_op.__name__ = str_op
                _cache_fake_ops_dict[(str(target), name)] = fake_op
                return fake_op
            else:
                target = getattr(target, name)
                _cache_ops_dict[(str(target), name)] = target
        return target

    with setting_python_recursive_limit():
        # @lint-ignore PYTHONPICKLEISBAD
        body = pickle.loads(state_bytes)

    # Get the target ops since we serialized the targets with just their name
    graph = body["_graph"]
    for node in graph.nodes:
        # Given the name of an operation, find the actual Op object
        # Ex. Given `aten.add.Tensor` we will return `OpOverload(op='aten.add', overload='Tensor')`
        if node.op == "call_function" and isinstance(node.target, str):
            node.target = str_to_op(node.target)

        if (original_aten := node.meta.get("original_aten", None)) is not None:
            node.meta["original_aten"] = str_to_op(original_aten)

        if (source_fn := node.meta.get("source_fn", None)) is not None:
            node.meta["source_fn"] = (source_fn[0], str_to_op(source_fn[1]))

        if (nn_module_stack := node.meta.get("nn_module_stack", None)) is not None:
            nn_module_stack = node.meta["nn_module_stack"]
            for key, val in nn_module_stack.items():
                nn_module_stack[key] = (val[0], str_to_op(val[1]))

    # fx.GraphModule constructor expects the totally flattened dictionary for attributes
    # but directly passing the module dict doesn't comply with that format. (some attributes are saved under `_buffers` in module dict etc)
    # So, we work around this by creating a dummy module which contains the original module's attributes
    root = torch.nn.Module()
    root.__dict__ = body

    gm = make_export_graph_module(root, graph, body["_graphmodule_cls_name"])
    meta = get_exir_meta(gm)
    meta.in_spec = body["_in_spec"]
    meta.out_spec = body["_out_spec"]
    gm.recompile()
    return gm


@dataclasses.dataclass
class ExirMetadata:
    """The fields in this class are what used to be extra data from ExportGraphModule."""

    in_spec: Optional[pytree.TreeSpec] = None
    out_spec: Optional[pytree.TreeSpec] = None
    update_spec: int = 0  # TODO more information here.
    # Mapping from output name to mutated buffer names.
    mutation: List[Tuple[str, List[str]]] = dataclasses.field(default_factory=list)


EXIR_METADATA = "_exir_metadata_key"


def get_exir_meta(gm: fx.GraphModule) -> ExirMetadata:
    if EXIR_METADATA not in gm.meta:
        raise AssertionError(
            "GraphModule does not have EXIR metadata associated with it."
        )
    return gm.meta[EXIR_METADATA]


def is_exir_graph_module(gm: fx.GraphModule) -> bool:
    return EXIR_METADATA in gm.meta


class ExportGraphModuleMixin:
    def __reduce__(self) -> Tuple[Callable[..., "ExportGraphModule"], Tuple[bytes]]:
        """
        Serialization of the ExportGraphModule. The FX serialization does not
        serialize the underlying graph to preserve backwards-compatiblity and
        instead retraces the graph module when loading.  This results in loss of
        metadata that is later used for optimizations directly on the graph
        module.  Since we want to preserve this metadata and we do not care that
        much about BC, we will write our own serialization method.
        """
        # pyre-ignore
        def op_to_str(op) -> str:
            try:
                pickle.dumps(op)
            except TypeError:
                if isinstance(op, torch._ops.HigherOrderOperator):
                    return f"torch.ops.{op.__name__}"
                elif "torch" in op.__module__:
                    return f"torch.ops.{str(op)}"
                else:
                    raise pickle.PickleError(f"Unable to pickle op {op}")
            except AttributeError:
                if "torch" in op.__module__:
                    return f"{str(op)}"
                else:
                    raise pickle.PickleError(f"Unable to pickle op {op}")
            return op

        gm_dict = self.__dict__.copy()
        gm_dict["_graphmodule_cls_name"] = self.__class__.__name__

        graph = copy.deepcopy(gm_dict["_graph"])
        for node in graph.nodes:
            # Replace the ops with their names since we cannot serialize the ops
            if node.op == "call_function":
                node.target = op_to_str(node.target)

            if (original_aten := node.meta.get("original_aten", None)) is not None:
                node.meta["original_aten"] = op_to_str(original_aten)

            if (source_fn := node.meta.get("source_fn", None)) is not None:
                node.meta["source_fn"] = (source_fn[0], op_to_str(source_fn[1]))

            if (nn_module_stack := node.meta.get("nn_module_stack", None)) is not None:
                nn_module_stack = node.meta["nn_module_stack"]
                for key, val in nn_module_stack.items():
                    nn_module_stack[key] = (val[0], op_to_str(val[1]))

            # Check if other metadata are pickleable
            unpickleable_keys = ["val"]
            for key, val in node.meta.items():
                if key in unpickleable_keys:
                    continue
                try:
                    pickle.dumps(val)
                except (pickle.PickleError, TypeError):
                    warnings.warn(
                        f"Cannot pickle node {node}'s metadata {key} with value {val}."
                    )
                    unpickleable_keys.append(key)

            for key in unpickleable_keys:
                if node.meta.get(key) is not None:
                    del node.meta[key]

        gm_dict["_graph"] = graph
        meta = self.meta[EXIR_METADATA]  # pyre-ignore
        gm_dict["_in_spec"] = copy.deepcopy(meta.in_spec)
        gm_dict["_out_spec"] = copy.deepcopy(meta.out_spec)
        with setting_python_recursive_limit():
            pickled_state = pickle.dumps(gm_dict)

        return (reduce_graph_module, (pickled_state,))

    # pyre-ignore
    def forward(self, *args: Any) -> Any:
        meta = self.meta[EXIR_METADATA]  # pyre-ignore
        if getattr(meta, "in_spec", None) is not None:
            try:
                # pyre-ignore
                args = fx_pytree.tree_flatten_spec(args, meta.in_spec)
            except Exception as e:
                raise InternalError("The in_spec is not correctly maintained.") from e

        with torch.fx.traceback.preserve_node_meta(), torch.no_grad():
            res = torch.fx.Interpreter(self).run(*args, enable_io_processing=False)

        if getattr(meta, "out_spec", None) is not None:
            try:
                mutation = meta.mutation
                num_mutated = len(mutation) if mutation is not None else 0
                res = pytree.tree_unflatten(
                    res[num_mutated:],
                    meta.out_spec,
                )
                return res
            except Exception as e:
                raise InternalError("The out_spec is not correctly maintained.") from e
        return res

    def recompile(self) -> torch.fx.graph_module.PythonCode:
        """
        Generates the code for this GraphModule from its ``graph`` attribute for
        testing (with FileCheck) and debugging purposes.
        """
        python_code = self._graph.python_code(root_module="self")  # pyre-ignore
        self._code = python_code.src  # pyre-ignore
        return python_code

    def get_export_graph(self) -> ExportGraph:
        assert isinstance(self, fx.GraphModule)
        return ExportGraph(self, self.graph)


def attach_export_graph_metadata(gm: fx.GraphModule, meta: ExirMetadata) -> None:
    gm.meta[EXIR_METADATA] = meta


def make_export_graph_module(
    root: Union[torch.nn.Module, Dict[str, Any]],
    graph: fx.Graph,
    class_name: str = "ExportGraphModule",
) -> fx.GraphModule:
    gm = fx.GraphModule(root, graph, class_name)
    meta = ExirMetadata(
        in_spec=None,
        out_spec=None,
        update_spec=0,
    )
    attach_export_graph_metadata(gm, meta)
    return gm


def _get_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, arg_index: int
) -> Tuple[str, torch.nn.Module, torch.fx.Node]:
    submod_node = node.args[arg_index]
    assert isinstance(submod_node, torch.fx.Node)
    assert submod_node.op == "get_attr"
    assert isinstance(submod_node.target, str)
    submodule = graph_module.get_submodule(submod_node.target)
    # pyre-ignore
    return submod_node.target, submodule, node


def get_control_flow_submodules(
    graph_module: ExportGraphModule,
) -> List[Tuple[str, ExportGraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target is torch.ops.cond:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))
            control_flow_submodules.append(_get_submodule(graph_module, node, 2))
        if node.target is torch.ops.map_impl:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))

    return control_flow_submodules
