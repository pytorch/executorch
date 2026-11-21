# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower native Helion kernel calls to ordinary Triton launches for AOTI.

AOTInductor cannot compile Helion's higher-order ops into a standalone C++
wrapper, but it can compile Triton kernels wrapped by ``wrap_triton``. This pass
renders each Helion call with its single, explicitly selected config, traces the
generated host code with fake tensors and splices the resulting Triton launch
into the exported program. Helion is imported only when a Helion call is found.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import importlib.metadata
import inspect
import operator
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, cast, NamedTuple

import torch
from torch._inductor.codecache import PyCodeCache
from torch._library.triton import set_wrap_triton_enabled
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_arg
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from triton.runtime.jit import JITFunction

# Launch options AOTI reproduces; anything else would be silently dropped.
# wrap_triton records launch kwargs by name on the captured kernel call, and
# Inductor materializes them as backend options for the AOTI compilation.
_SUPPORTED_LAUNCH_OPTIONS = ("num_warps", "num_stages", "maxnreg", "enable_fp_fusion")

# Helion has no public API for a call that torch.export captured: the HOPs, their
# side table and their output spec belong to Helion's private torch.compile
# integration, and so does the launcher parameter of a standalone entrypoint. The
# pass reads that contract only through this section and otherwise uses Helion's
# public API (Kernel.bind, BoundKernel.to_code). The one other private trait the
# pass relies on is the functional HOP's (outputs, {name: clone}) return layout,
# which _make_hop_replacement mirrors and _reroute_hop_outputs indexes into.
_HELION_HOP_NAMES = (
    "helion_kernel_wrapper_mutation",
    "helion_kernel_wrapper_functional",
)
_LAUNCHER_PARAMETER = "_launcher"
_HELION_RUNTIME_PREFIX = "helion."
_HELION_REQUIREMENT = "Helion 1.5.2 or newer"
_MIN_HELION_VERSION = (1, 5, 2)


class _CapturedHelionCall(NamedTuple):
    kernel: Any
    is_functional: bool
    tensor_args: Mapping[str, Node]
    has_constant_args: bool
    tensors_to_clone: tuple[str, ...]
    tensor_output_count: int
    aliases_inputs: bool
    mutates_inputs: bool
    output_spec: Mapping[str, Any]


def _check_helion_version() -> None:
    try:
        version: str | None = importlib.metadata.version("helion")
    except importlib.metadata.PackageNotFoundError:
        # A Helion importable without distribution metadata (for example a
        # source checkout) may still name its version on the module. When no
        # version is discoverable, the attribute probes in
        # _helion_capture_modules are the version evidence available.
        helion = importlib.import_module("helion")
        version = getattr(helion, "__version__", None)
    if version is None:
        return
    parsed = []
    for part in version.split(".")[:3]:
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            return
        parsed.append(int(digits))
    if tuple(parsed) < _MIN_HELION_VERSION:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"found Helion {version}"
        )


@functools.cache
def _helion_capture_modules() -> tuple[ModuleType, ModuleType]:
    _check_helion_version()
    try:
        hops = importlib.import_module("helion._compiler._dynamo.higher_order_ops")
        variables = importlib.import_module("helion._compiler._dynamo.variables")
    except ImportError as error:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}"
        ) from error
    required = (
        (hops, "helion_kernel_side_table"),
        (hops, "helion_kernel_wrapper_functional"),
        (variables, "_get_flat_output"),
    )
    missing = [f"{m.__name__}.{name}" for m, name in required if not hasattr(m, name)]
    if missing:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"this Helion lacks {', '.join(missing)}"
        )
    return hops, variables


def _read_captured_call(node: Node) -> _CapturedHelionCall:
    hops, _ = _helion_capture_modules()
    kwargs = cast(Mapping[str, Any], node.kwargs)
    # Read the private fields strictly: a Helion that renames one must fail
    # with the requirement error instead of surfacing a bare KeyError. The
    # two exceptions are direct_aliases and mutated_inputs: Helion omits them
    # from a capture its functionalization rebuilt when they are empty, so a
    # missing entry means "none". The traced replacement is checked for
    # writes to its inputs and for outputs aliasing them either way, so a
    # capture that understates either is still rejected after tracing.
    try:
        spec = kwargs["output_spec"]
        is_functional = node.target is hops.helion_kernel_wrapper_functional
        return _CapturedHelionCall(
            kernel=hops.helion_kernel_side_table.get_kernel(kwargs["kernel_idx"]),
            is_functional=is_functional,
            tensor_args=kwargs["tensor_args"],
            has_constant_args=bool(kwargs["constant_args"]),
            tensors_to_clone=(
                tuple(kwargs["tensors_to_clone"]) if is_functional else ()
            ),
            tensor_output_count=sum(
                leaf["type"] == "tensor" for leaf in spec["leaf_specs"]
            ),
            aliases_inputs=bool(spec.get("direct_aliases")),
            mutates_inputs=bool(spec.get("mutated_inputs")),
            output_spec=spec,
        )
    except KeyError as error:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"this capture lacks the expected field {error}"
        ) from error


def _check_capture_semantics(bound: Any, captured: _CapturedHelionCall) -> None:
    """Reject a kernel whose host code or environment changed since export."""
    _, variables = _helion_capture_modules()
    expected = captured.output_spec.get("_host_semantic_fingerprint")
    if expected is None or not hasattr(bound, "_get_host_semantic_fingerprint"):
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}, which records "
            "host-semantic fingerprints"
        )
    leaves, _, _ = variables._get_flat_output(bound.host_function)
    normalization = captured.output_spec.get("_host_semantic_input_normalization", ())
    actual = bound._get_host_semantic_fingerprint(
        leaves, input_normalization=normalization
    )
    if actual != expected:
        raise RuntimeError(
            "Helion capture and CUDA lowering have different host semantics"
        )


# ---------------------------------------------------------------------------


class _HelionCall(NamedTuple):
    captured: _CapturedHelionCall
    argument_names: list[str]
    argument_nodes: tuple[Node, ...]
    fake_arguments: tuple[torch.Tensor, ...]


def _check_single_launch_site(source: str, entrypoint: str) -> None:
    module = ast.parse(source)
    entry = next(
        (
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == entrypoint
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Generated Triton source has no entrypoint {entrypoint!r}")
    launches = [
        n
        for n in ast.walk(entry)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == _LAUNCHER_PARAMETER
    ]
    if len(launches) != 1:
        raise NotImplementedError(
            "Helion lowering requires a single Triton launch site"
        )
    for keyword in launches[0].keywords:
        if keyword.arg in _SUPPORTED_LAUNCH_OPTIONS:
            continue
        if (
            keyword.arg == "launch_cooperative_grid"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
        ):
            continue
        raise NotImplementedError(
            f"Helion lowering does not support launch option {keyword.arg!r}; "
            "the CUDA/AOTI runtime must preserve its launch contract first"
        )


def _launch_through_wrap_triton(kernel, grid, *args, **kwargs):
    return torch.library.wrap_triton(kernel)[grid](*args, **kwargs)


def _parse_helion_call(captured: _CapturedHelionCall) -> _HelionCall:
    kernel = captured.kernel
    if kernel.settings.backend != "triton":
        raise NotImplementedError(
            "Helion lowering supports only the Triton backend; this kernel uses "
            f"{kernel.settings.backend!r}"
        )
    if not kernel.settings.static_shapes:
        raise NotImplementedError("Helion lowering requires static_shapes=True")
    if len(kernel.configs) != 1:
        raise ValueError(
            "Select one Helion config offline with @helion.kernel(config=...) before CUDA lowering"
        )
    if captured.aliases_inputs:
        raise NotImplementedError(
            "Helion lowering does not yet support aliased outputs"
        )
    if not captured.is_functional and captured.mutates_inputs:
        raise NotImplementedError(
            "Mutating Helion calls must be functionalized before CUDA lowering"
        )
    names = list(kernel.signature.parameters)
    if captured.has_constant_args or set(names) != set(captured.tensor_args):
        raise NotImplementedError(
            "Helion lowering currently requires tensor-only arguments"
        )
    nodes = tuple(captured.tensor_args[name] for name in names)
    call = _HelionCall(captured, names, nodes, tuple(n.meta["val"] for n in nodes))
    _check_static_arguments(call)
    return call


def _check_static_arguments(call: _HelionCall) -> None:
    for name, value in zip(call.argument_names, call.fake_arguments):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise NotImplementedError("Helion lowering requires strided tensor inputs")
        if any(not isinstance(size, int) for size in (*value.shape, *value.stride())):
            raise NotImplementedError(
                "Helion lowering does not yet preserve dynamic specialization guards"
            )
        # The kernel is specialized to the strides of the uncloned argument.
        if (
            name in call.captured.tensors_to_clone
            and value.clone().stride() != value.stride()
        ):
            raise NotImplementedError(
                "Helion lowering does not support mutation clones that change strides"
            )


def _bind_helion_jit_helpers(source: str) -> str:
    """Call Helion's in-kernel Triton helpers through global names.

    Generated kernels call helpers such as ``helion_triton_helpers.<helper>``.
    When AOTInductor compiles a user Triton kernel it copies the JIT functions
    the kernel references by global name, but not those reached through a
    module, so each such helper gets a global binding of its own name.
    """
    namespace = vars(PyCodeCache.load(source))
    tree = ast.parse(source)
    module_of: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        module = namespace.get(node.value.id)
        if not (
            inspect.ismodule(module)
            and module.__name__.startswith(_HELION_RUNTIME_PREFIX)
            and isinstance(getattr(module, node.attr, None), JITFunction)
        ):
            continue
        if node.attr in namespace or hasattr(builtins, node.attr):
            raise NotImplementedError(
                f"Helion helper {node.attr!r} clashes with a name in the generated module"
            )
        module_of[node.attr] = node.value.id
    if not module_of:
        return source

    class _Unqualify(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if (
                isinstance(node.value, ast.Name)
                and module_of.get(node.attr) == node.value.id
            ):
                return ast.copy_location(ast.Name(node.attr, ast.Load()), node)
            return self.generic_visit(node)

    tree = _Unqualify().visit(tree)
    for name, module_name in module_of.items():
        index = next(
            i
            for i, stmt in enumerate(tree.body)
            if isinstance(stmt, (ast.Import, ast.ImportFrom))
            and any((alias.asname or alias.name) == module_name for alias in stmt.names)
        )
        tree.body.insert(index + 1, ast.parse(f"{name} = {module_name}.{name}").body[0])
    return ast.unparse(ast.fix_missing_locations(tree))


def _render_triton_entrypoint(call: _HelionCall) -> Callable[..., Any]:
    kernel = call.captured.kernel
    config = kernel.configs[0]
    if config.get("host_tensor_descriptors", False):
        raise NotImplementedError(
            "Helion lowering does not support host tensor descriptors"
        )
    bound = kernel.bind(call.fake_arguments)
    _check_capture_semantics(bound, call.captured)
    # The default output keeps Helion's runtime imports; with
    # allow_helion_deps=False Helion drops the import of its in-kernel helpers
    # without inlining them. AOTInductor compiles the kernels, so inference
    # still needs no Helion.
    source = bound.to_code(config)
    _check_single_launch_site(source, kernel.name)
    return getattr(PyCodeCache.load(_bind_helion_jit_helpers(source)), kernel.name)


def _make_hop_replacement(
    call: _HelionCall, entrypoint: Callable[..., Any]
) -> Callable[..., Any]:
    """Call the Triton entrypoint and return outputs shaped like the Helion HOP's."""
    captured = call.captured

    def run(*values):
        arguments = dict(zip(call.argument_names, values))
        cloned = {name: arguments[name].clone() for name in captured.tensors_to_clone}
        launch_args = (
            cloned.get(name, arguments[name]) for name in call.argument_names
        )
        outputs = entrypoint(
            *launch_args, **{_LAUNCHER_PARAMETER: _launch_through_wrap_triton}
        )
        flat = tuple(
            leaf
            for leaf in pytree.tree_leaves(outputs)
            if isinstance(leaf, torch.Tensor)
        )
        if len(flat) != captured.tensor_output_count:
            raise RuntimeError("Helion lowering changed the captured output structure")
        return (flat, cloned) if captured.is_functional else flat

    return run


def _check_no_placeholder_mutation(traced: GraphModule, kernel_name: str) -> None:
    placeholders = {n for n in traced.graph.nodes if n.op == "placeholder"}
    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        for index, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            if index < len(node.args):
                value = node.args[index]
            elif argument.name in node.kwargs:
                value = node.kwargs[argument.name]
            else:
                continue
            if any(leaf in placeholders for leaf in pytree.tree_leaves(value)):
                raise NotImplementedError(
                    f"Helion lowering traced a write to an input of kernel "
                    f"{kernel_name!r} that Helion did not report as mutated; "
                    "splicing it would mutate a graph input with no matching "
                    "graph-signature entry"
                )


def _trace_helion_call_as_triton(node: Node) -> tuple[GraphModule, tuple[Node, ...]]:
    captured = _read_captured_call(node)
    call = _parse_helion_call(captured)
    run = _make_hop_replacement(call, _render_triton_entrypoint(call))
    with set_wrap_triton_enabled(True):
        traced = make_fx(torch.func.functionalize(run), tracing_mode="fake")(
            *call.fake_arguments
        )
    if not any("triton_kernel_wrapper" in str(n.target) for n in traced.graph.nodes):
        raise RuntimeError("Helion lowering lost the Triton launch during capture")
    _check_no_placeholder_mutation(traced, captured.kernel.name)
    _check_outputs_do_not_alias_inputs(traced)
    return traced, call.argument_nodes


def _check_outputs_do_not_alias_inputs(traced: GraphModule) -> None:
    inputs = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in traced.graph.nodes
        if n.op == "placeholder"
    }
    for output in pytree.tree_leaves(traced.graph.output_node().args[0]):
        if (
            isinstance(output, Node)
            and StorageWeakRef(output.meta["val"].untyped_storage()) in inputs
        ):
            raise NotImplementedError(
                "Helion lowering does not yet support outputs that alias inputs"
            )


def _splice_traced_graph(
    graph: Graph, before: Node, traced: GraphModule, inputs: tuple[Node, ...]
) -> Any:
    placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
    environment = dict(zip(placeholders, inputs))
    result = None
    with graph.inserting_before(before):
        for traced_node in traced.graph.nodes:
            if traced_node.op == "placeholder":
                continue
            if traced_node.op == "output":
                result = map_arg(traced_node.args[0], environment.__getitem__)
            elif traced_node.op == "call_function":
                copied = graph.node_copy(traced_node, environment.__getitem__)
                # Keep the Helion call's provenance on the replacement region;
                # node_copy alone keeps only the make_fx metadata.
                for key in ("stack_trace", "nn_module_stack", "from_node"):
                    if key in before.meta:
                        copied.meta[key] = before.meta[key]
                environment[traced_node] = copied
            else:
                raise NotImplementedError(
                    f"Helion lowering cannot inline {traced_node.op} nodes"
                )
    return result


def _reroute_hop_outputs(
    graph: Graph, node: Node, replacement: Any, signature: ExportGraphSignature | None
) -> None:
    """Replace ``node`` and its ``getitem`` users with the matching replacement nodes."""
    if isinstance(replacement, Node):
        _check_same_tensor_layout(node, replacement)
        node.replace_all_uses_with(replacement)
        if signature is not None:
            signature.replace_all_uses(node.name, replacement.name)
    else:
        for user in list(node.users):
            if user.op != "call_function" or user.target is not operator.getitem:
                raise NotImplementedError(
                    "Helion lowering requires captured outputs to be accessed by getitem"
                )
            _reroute_hop_outputs(graph, user, replacement[user.args[1]], signature)
    graph.erase_node(node)


def _check_same_tensor_layout(original: Node, replacement: Node) -> None:
    # Export traces Helion outputs as contiguous fakes; code after the call may
    # have specialized on that layout.
    expected, actual = original.meta.get("val"), replacement.meta["val"]
    if not isinstance(expected, torch.Tensor):
        raise NotImplementedError(
            f"Helion lowering replaced node {original.name!r}, whose exported "
            "value is missing or not a tensor, with a tensor; the captured "
            "output structure does not match what the pass expected"
        )
    if (expected.shape, expected.stride(), expected.dtype) != (
        actual.shape,
        actual.stride(),
        actual.dtype,
    ):
        raise NotImplementedError(
            f"Helion lowering produced {tuple(actual.shape)}/{actual.stride()}/{actual.dtype} for an output "
            f"exported as {tuple(expected.shape)}/{expected.stride()}/{expected.dtype}"
        )


class LowerHelionToTritonPass:
    """Replace native Helion calls with the Triton launches of their selected config.

    Supported: one explicit config per kernel, static strided tensor arguments,
    multiple tensor outputs and functionalized input mutation. Config selection
    stays offline; the pass neither tunes nor launches kernels, and inference
    needs neither Python nor Helion.
    """

    requires_exported_program = True

    def __call__(self, exported_program: ExportedProgram) -> None:
        root = exported_program.graph_module
        for module in root.modules():
            if not isinstance(module, GraphModule):
                continue
            signature = exported_program.graph_signature if module is root else None
            modified = False
            for node in list(module.graph.nodes):
                if (
                    node.op != "call_function"
                    or str(node.target) not in _HELION_HOP_NAMES
                ):
                    continue
                try:
                    traced, inputs = _trace_helion_call_as_triton(node)
                    replacement = _splice_traced_graph(
                        module.graph, node, traced, inputs
                    )
                    _reroute_hop_outputs(module.graph, node, replacement, signature)
                except Exception as error:
                    kernel_name = "unknown"
                    try:
                        kernel_name = _read_captured_call(node).kernel.name
                    except Exception:
                        pass
                    error.add_note(
                        f"while lowering Helion kernel {kernel_name!r} "
                        f"(node {node.name})"
                    )
                    raise
                modified = True
            if modified:
                module.graph.lint()
                module.recompile()
